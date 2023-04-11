import findspark

findspark.init()

from pyspark.sql import SparkSession
from numba import cuda


def check_gpu_availability():
    try:
        num_gpus = len(cuda.gpus)
        if num_gpus > 0:
            print("GPU(s) detected:")
            for i, gpu in enumerate(cuda.gpus):
                with gpu:
                    print(f"  GPU {i}: {cuda.current_context().get_memory_info()}")
            return True
        else:
            print("No GPU detected.")
            return False
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("GPU Availability Test") \
        .getOrCreate()

    is_gpu_available = check_gpu_availability()

    if is_gpu_available:
        print("You can use GPU with PySpark.")
    else:
        print("You cannot use GPU with PySpark.")

    spark.stop()
