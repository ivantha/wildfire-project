from pyspark import SparkConf, SparkContext

# Create a Spark configuration object
conf = SparkConf()

# Create a Spark context object
sc = SparkContext(conf=conf)

# Get the name of the Spark application
app_name = sc.appName

# Get the master URL of the Spark cluster
master_url = sc.master

# Get the Spark version
spark_version = sc.version

# Get the number of worker nodes in the Spark cluster
num_nodes = sc.defaultParallelism

# Print out the information about the cluster
print(f"Spark application name: {app_name}")
print(f"Spark master URL: {master_url}")
print(f"Spark version: {spark_version}")
print(f"Number of worker nodes: {num_nodes}")
