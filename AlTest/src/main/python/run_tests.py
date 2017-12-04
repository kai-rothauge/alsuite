#!/usr/bin/env python

import argparse
import imp
import os
import logging

from __init__ import PROJ_DIR, TEST_DIR
from commands import *
from testsuites import *


logger = logging.getLogger("alchemistperf")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(description='Run Alchemist performance tests. Before running, '
    'edit the supplied configuration file.')

parser.add_argument('--config-file', help='override default location of config file, must be a '
    'python file that ends in .py', default="%s/config/config.py" % TEST_DIR)

parser.add_argument('--additional-make-distribution-args',
    help='additional arugments to pass to make-distribution.sh when building Spark', default="")

args = parser.parse_args()
assert args.config_file.endswith(".py"), "config filename must end with .py"

# Check if the config file exists.
assert os.path.isfile(args.config_file), ("Please create a config file called %s (you probably "
    "just want to copy and then modify %s/config/config.py.template)" %
    (args.config_file, TEST_DIR))

print("Detected project directory: %s" % PROJ_DIR)
# Import the configuration settings from the config file.
print("Loading configuration from %s" % args.config_file)
with open(args.config_file) as cf:
    config = imp.load_source("config", "", cf)
    
run_tests = config.RUN_TESTS and (len(config.TESTS) > 0)

# Only build the perf test sources that will be used.
should_prep_tests = run_tests and config.PREP_TESTS

# # Restart Master and Workers
# should_restart_cluster = config.RESTART_SPARK_CLUSTER
# # Copy all the files in SPARK_HOME 
# should_rsync_spark_home = config.RSYNC_SPARK_HOME

# # Check that commit ID's are specified in config_file.
# if should_prep_spark:
#     assert config.SPARK_COMMIT_ID is not "", \
#         ("Please specify SPARK_COMMIT_ID in %s" % args.config_file)
# 
# # If a cluster is already running from the Spark EC2 scripts, try shutting it down.
# if os.path.exists(config.SPARK_HOME_DIR) and should_restart_cluster and not config.IS_MESOS_MODE:
#     Cluster(spark_home=config.SPARK_HOME_DIR).stop()
# 
# spark_build_manager = SparkBuildManager("%s/spark-build-cache" % PROJ_DIR, config.SPARK_GIT_REPO)
# 
# if config.IS_MESOS_MODE:
#     cluster = MesosCluster(spark_home=config.SPARK_HOME_DIR, spark_conf_dir=config.SPARK_CONF_DIR,
#                            mesos_master=config.SPARK_CLUSTER_URL)
# elif config.USE_CLUSTER_SPARK:
#     cluster = Cluster(spark_home=config.SPARK_HOME_DIR, spark_conf_dir=config.SPARK_CONF_DIR)
# else:
#     cluster = spark_build_manager.get_cluster(
#         commit_id=config.SPARK_COMMIT_ID,
#         conf_dir=config.SPARK_CONF_DIR,
#         merge_commit_into_master=config.SPARK_MERGE_COMMIT_INTO_MASTER,
#         is_yarn_mode=config.IS_YARN_MODE,
#         additional_make_distribution_args=args.additional_make_distribution_args)
# 
# # rsync Spark to all nodes in case there is a change in Worker config
# if should_restart_cluster and should_rsync_spark_home:
#     cluster.sync_spark()

# # If a cluster is already running from an earlier test, try shutting it down.
# if os.path.exists(cluster.spark_home) and should_restart_cluster:
#     cluster.stop()
# 
# if should_restart_cluster:
#     # Ensure all shutdowns have completed (no executors are running).
#     cluster.ensure_spark_stopped_on_slaves()

print("Building Alchemist Tests")
Tests.initialize(config)
if should_prep_tests:
    Tests.build()
elif run_tests:
    assert Tests.is_built(), ("You tried to skip packaging the Alchemist perf " +
        "tests, but %s was not already present") % Tests.test_jar_path

if run_tests:
    Tests.run()

# if should_restart_cluster:
#     cluster.stop()

print("Finished running all tests.")
