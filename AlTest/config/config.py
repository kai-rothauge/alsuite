"""
Configuration options for running Alchemist performance tests.
"""

import time
import os
import os.path
import socket

from __init__ import PROJ_DIR, CONFIG_DIR
from config_utils import Setting, FlagSet, JavaOptionSet, OptionSet, ConstantOption


# =========================================================== #
#                Standard Configuration Options               #
# =========================================================== #

# Which tests to run
RUN_TESTS = True

# Set this to true for the first installation or whenever you make a change to the tests.
PREP_TESTS = True

# Version number of Alchemist
ALTEST_VERSION = "0.1"

# =========================================================== #
#                  Spark Configuration Options                #
# =========================================================== #

TEST_JAR_PATH = "%s/target/scala-2.11/altest-assembly-%s.jar" % (PROJ_DIR, ALTEST_VERSION)

if os.getenv('NERSC_HOST') != None:             # If running on Cori
    SPARK_DRIVER_MEMORY   = "5g"
    SPARK_EXECUTOR_MEMORY = "5g"
    SPARK_NUM_EXECUTORS   = "2"
    
    SPARK_SETTINGS = [Setting("driver-memory",SPARK_DRIVER_MEMORY), Setting("executor-memory",SPARK_EXECUTOR_MEMORY), Setting("num-executors",SPARK_NUM_EXECUTORS)]
else:                                           # If running Spark locally:
    SPARK_CLUSTER_URL     = "local[3]"
    
    SPARK_SETTINGS = [Setting("master",SPARK_CLUSTER_URL)]

# SPARK_VERSION = 1.5


# =========================================================== #
#                 Output Configuration Options                #
# =========================================================== #

# Directory in project folder to write results to
OUTPUT_DIR = PROJ_DIR + "/results/" 
# OUTPUT_DIR = PROJ_DIR + ("/results/test_output_%s" % (time.strftime("%Y-%m-%d_%H-%M-%S")))

# Console output color (ANSI colors only)
# Options: Black, Red, Green, Yellow, Blue, Magenta, Cyan, White
#          Bright Black, Bright Red, Bright Green, Bright Yellow,
#          Bright Blue, Bright Magenta, Bright Cyan, Bright White
# CONSOLE_COLOR_MAIN = "bright cyan"
CONSOLE_COLOR = "white"       
                                    
LOG_LEVEL = "all"                   # Options: all, none, error

OUTPUT_SETTINGS = [Setting("out-dir",OUTPUT_DIR), Setting("console-color",CONSOLE_COLOR), Setting("log-level",LOG_LEVEL)]

# =========================================================== #
#                   Test Configuration Options                #
# =========================================================== #

# Set up OptionSets. Note that giant cross product is done over all JavaOptionsSets + OptionSets
# passed to each test which may be combinations of those set up here.

# Java options.
COMMON_JAVA_OPTS = [
#     # Fraction of JVM memory used for caching RDDs.
#     JavaOptionSet("spark.storage.memoryFraction", [0.66]),
#     JavaOptionSet("spark.serializer", ["org.apache.spark.serializer.JavaSerializer"]),
#     # JavaOptionSet("spark.executor.memory", ["9g"]),
#     # Turn event logging on in order better diagnose failed tests. Off by default as it crashes
#     # releases prior to 1.0.2
#     # JavaOptionSet("spark.eventLog.enabled", [True]),
#     # To ensure consistency across runs, we disable delay scheduling
#     JavaOptionSet("spark.locality.wait", [str(60 * 1000 * 1000)])
]

# The following options value sets are shared among all tests.
COMMON_OPTS = [
    # How many times to run each experiment - used to warm up system caches.
    # This OptionSet should probably only have a single value (i.e., length 1)
    # since it doesn't make sense to have multiple values here.
    OptionSet("num-trials", [1]),
    # Extra pause added between trials, in seconds. For runs with large amounts
    # of shuffle data, this gives time for buffer cache write-back.
    OptionSet("inter-trial-wait", [1])
]

# =========================================================== #
#                           Test Setup                        #
# =========================================================== #

# Clustering tests
RUN_KMEANS_TEST              = False

# Linear algebra tests
RUN_SVD_TEST                 = False
RUN_MATRIX_MULT_TEST         = True


# ------------------------------------------------------------#

RUN_CLUSTERING_TEST = [RUN_KMEANS_TEST]
RUN_LIN_ALG_TEST = [RUN_SVD_TEST or RUN_MATRIX_MULT_TEST]

# ------------------------------------------------------------#

TESTS = []
#PERF_TEST_RUNNER = "alchemist.TestRunner"

# ------------------------------------------------------------#

# The following options value sets are shared among all tests of 
# operations on Alchemist algorithms.
COMMON_OPTS = COMMON_OPTS + [
    # The number of input partitions
    OptionSet("num-partitions", [128]),
    # A random seed to make tests reproducible
    OptionSet("random-seed", [5])
]

# Clustering Tests
if RUN_CLUSTERING_TEST:
    if RUN_KMEANS_TEST:
        
        GENERATE_DATA = False

        SHORT_NAME  = "kmeans"
        LONG_NAME   = "K-Means Clustering"
        KMEANS_META = [Setting("short-name", SHORT_NAME), Setting("long-name", LONG_NAME), Setting("generate-data", str(GENERATE_DATA))]

        KMEANS_TEST_OPTS = COMMON_OPTS + [
            OptionSet("num-centers", [10]),                 # The number of centers
            OptionSet("num-iterations", [100]),             # The number of iterations for KMeans
            OptionSet("change-threshold", [1e-4])           # Change threshold for cluster centers
        ]
        if GENERATE_DATA:
            KMEANS_TEST_OPTS = KMEANS_TEST_OPTS + [
                OptionSet("num-examples", [1000000]),       # The number of examples
                OptionSet("num-features", [10000]),         # The number of features per point
            ]
        else:
            KMEANS_TEST_OPTS = KMEANS_TEST_OPTS + [
                OptionSet("data-file", [PROJ_DIR + "/data/mnist.t"])
            ]
        
        TESTS += [(KMEANS_META, KMEANS_TEST_OPTS)]
        

if RUN_LIN_ALG_TEST:
    if RUN_SVD_TEST:
        
        GENERATE_DATA = False
        
        SHORT_NAME  = "svd"
        LONG_NAME   = "SVD"
        SVD_META = [Setting("short-name", SHORT_NAME), Setting("long-name", LONG_NAME), Setting("generate-data", str(GENERATE_DATA))]

        SVD_TEST_OPTS = COMMON_OPTS + [
            OptionSet("rank", [10])                 # The number of top singular values wanted
        ]
        if GENERATE_DATA:
            SVD_TEST_OPTS = SVD_TEST_OPTS + [
                OptionSet("num-rows", [1000]),          # The number of rows for the matrix
                OptionSet("num-cols", [500])           # The number of columns for the matrix
            ]
        else:
            SVD_TEST_OPTS = SVD_TEST_OPTS + [
                OptionSet("data-file", [PROJ_DIR + "/data/mnist.t"])
            ]    
        
        TESTS += [(SVD_META, SVD_TEST_OPTS)]


    if RUN_MATRIX_MULT_TEST:
        
        GENERATE_DATA = True
        
        SHORT_NAME  = "matmult"
        LONG_NAME   = "Matrix Multiply"
        MATRIX_MULT_META = [Setting("short-name", SHORT_NAME), Setting("long-name", LONG_NAME), Setting("generate-data", str(GENERATE_DATA))]
        
        if GENERATE_DATA:
            MATRIX_MULT_TEST_OPTS = COMMON_OPTS + [
                OptionSet("M", [5000]),              # Number of rows of matrix A
                OptionSet("K", [2000]),              # Number of columns of matrix A/rows of matrix B
                OptionSet("N", [5000]),              # Number of columns of matrix B
                OptionSet("scale-A", [10]),         # Scaling parameter for matrix A
                OptionSet("scale-B", [1])           # Scaling parameter for matrix B
            ]
        else:
            MATRIX_MULT_TEST_OPTS = COMMON_OPTS + [
                OptionSet("data-file", [PROJ_DIR + "/data/mnist.t"])
            ]
    
        TESTS += [(MATRIX_MULT_META, MATRIX_MULT_TEST_OPTS)]
        