from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to process the 'frp' column
@udf(returnType=FloatType())
def process_frp(frp):
    values = list(map(float, frp.split(',')))
    return sum(values) / len(values)