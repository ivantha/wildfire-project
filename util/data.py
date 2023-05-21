from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to process the 'frp' column
@udf(returnType=FloatType())
def process_value_list_str(value_list_str):
    if value_list_str is None:
        return None
    else:
        values = list(map(float, value_list_str.split(',')))
        return sum(values) / len(values)
