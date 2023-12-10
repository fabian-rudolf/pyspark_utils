from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as fn
from pyspark.sql.window import Window
from typing import Optional

def calculate_central_tendency(df: DataFrame, value_column: str, key_columns: list) -> DataFrame:
    """
    Calculate measurements of central tendency for a given value column and key columns in a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.
    value_column (str): Name of the column containing the values for which to calculate the central tendency.
    key_columns (list): List of column names to group by.

    Returns:
    DataFrame: DataFrame with the calculated measurements of central tendency.
    """

    # Calculate mean, median (approximate), and mode
    stats_df = df.groupBy(key_columns).agg(
        fn.mean(value_column).alias('mean'),
        fn.expr(f'percentile_approx({value_column}, 0.5)').alias('median'),
        fn.expr(f'percentile_approx({value_column}, array(0.25, 0.75))').alias('iqr'),
        fn.expr(f'percentile_approx({value_column}, array(0.05, 0.95))').alias('ci_90')
    )

    # Calculate mode separately because it requires a different aggregation
    mode_df = df.groupBy(key_columns + [value_column]).count()
    window = Window.partitionBy(key_columns).orderBy(fn.desc('count'))
    mode_df = mode_df.withColumn('rank', fn.rank().over(window)).filter('rank = 1').drop('rank', 'count')

    # Join the mode DataFrame with the stats DataFrame
    result_df = stats_df.join(mode_df, on=key_columns, how='left')

    return result_df

def check_key_duplicates(spark: SparkSession, table_name: str, key_columns: list, limit: Optional[int] = 10):
    """
    Function to check the amount of key duplicates over columns in a table.
    
    Args:
    spark (SparkSession): Spark Session.
    table_name (str): Name of the table.
    key_columns (list): List of key columns.
    limit (int, optional): Limit for the output DataFrame if not `None`, defaults to 10.
    
    Returns:
    DataFrame: DataFrame with the amount and percentage of duplicates in data.
    """
    # Define window
    window = Window.partitionBy(*key_columns).orderBy(fn.lit(1))
    
    # Load table
    df = spark.table(table_name)
    
    # Calculate key duplicates
    df = df.withColumn('key_duplicate', fn.row_number().over(window))
    
    # Filter duplicates
    df = df.filter(fn.col('key_duplicate') > 1)
    
    # Group by key columns and calculate max duplicates
    df = df.groupBy(*key_columns).agg(fn.max('key_duplicate').alias('key_duplicates'))
    
    # Order by key duplicates
    df = df.orderBy(fn.desc('key_duplicates'))
    
    # Limit output
    if limit:
      df = df.limit(limit)
    
    return df
