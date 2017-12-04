import itertools
import os
import shutil
import sys
import json
from subprocess import Popen, PIPE

from __init__ import PROJ_DIR, TEST_DIR
from commands import run_cmd, SBT_CMD
from utils import OUTPUT_DIVIDER_STRING, append_config_to_file, stats_for_results


test_env = os.environ.copy()
TEST_DIR = PROJ_DIR + "/test"

class Tests(object):
    
    test_jar_path = "/path/to/test/jar"
    config = ''
        
    @classmethod
    def initialize(cls, config_):
        cls.config        = config_
        cls.test_jar_path = cls.config.TEST_JAR_PATH
        
    @classmethod
    def build(cls):
#         run_cmd("cd %s/test" % PROJ_DIR)
#         run_cmd("%s -Dspark.version=%s.0 clean assembly" % (SBT_CMD, spark_version))
        run_cmd("cd %s/.." % TEST_DIR, 0)
        run_cmd("sbt -build assembly", 0)
        run_cmd("cd %s" % TEST_DIR, 0)

    @classmethod
    def is_built(cls):
        """
        :return: True if this test suite has been built / compiled.
        """
        return os.path.exists(cls.test_jar_path)

    @classmethod
    def before_run_tests(cls):
        """
        This is called before tests in this suite are run.
        It is useful for logging test suite specific messages.

        :param out_file: a python file handler to the output file
        """
        pass

    @classmethod
    def run(cls):
        """
        Run a set of tests from this performance suite.
        """    
        tests_to_run = cls.config.TESTS                     # A list of 5-tuple elements specifying the tests to run.  See the
                                                        # 'Test Setup' section in config.py.template for more info.
        test_group_name = "Alchemist Tests"             # A short string identifier for this test run.
        output_dir = cls.config.OUTPUT_DIR                  # The output file where we write results.
        
        try:
            os.makedirs(output_dir,0o777)
        except:
            pass
        num_tests_to_run = len(tests_to_run)

        print(OUTPUT_DIVIDER_STRING)
        if num_tests_to_run == 1:
            print("Running %d test in %s" % (num_tests_to_run, test_group_name))
        else:
            print("Running %d tests in %s" % (num_tests_to_run, test_group_name))
        failed_tests = []

        cls.before_run_tests()
        
        spark_settings = []
        for i in cls.config.SPARK_SETTINGS:
            spark_settings.append(i.to_array()[0])
            
        output_settings = []
        for i in cls.config.OUTPUT_SETTINGS:
            output_settings.append(i.to_array()[0])
        
        main_class = "alchemist.TestRunner"

        for meta_data, opt_sets in tests_to_run:
            print(OUTPUT_DIVIDER_STRING + '\n')
#             print("Running test command: '%s' ... " % main_class)
            
            meta = {}
            meta_pairs = [i.to_tuple() for i in meta_data]
            for mp in meta_pairs:
                meta[mp[0].replace('-', '_')] = mp[1].replace('0x20', ' ')
                
            meta_settings = []
            for i in meta_data:
                meta_settings.append(i.to_array()[0])
            
#             stdout_filename = "%s/%s.out" % (output_dir, meta['short_name'])
#             stderr_filename = "%s/%s.err" % (output_dir, meta['short_name'])
#             
#             out_file = open(output_dir + "/" + meta['short_name'] + ".out", 'w')

            # Run a test for all combinations of the OptionSets given, then capture
            # and print the output.
            opt_set_arrays = [i.to_array() for i in opt_sets]
            for opt_list in itertools.product(*opt_set_arrays):

                cmd = cls.get_spark_submit_cmd(spark_settings, main_class, output_settings, meta_settings, opt_list)
#                         print("\nSetting env var SPARK_SUBMIT_OPTS: %s" % java_opts_str)
#                         test_env["SPARK_SUBMIT_OPTS"] = java_opts_str
                print("Running command:")
                print("%s\n" % cmd)
                Popen(cmd, shell=True, env=test_env).wait()

                try:
                    src = output_dir + meta['short_name'] + '_latest/'
                    src_files = os.listdir(src)
                    src_file = src_files[0][:-4]
                    new_dir  = output_dir + src_file
                    os.makedirs(new_dir)
                    for file_name in src_files:
                        full_file_name = os.path.join(src, file_name)
                        if (os.path.isfile(full_file_name)):
                            shutil.copy(full_file_name, new_dir)
                except:
                    pass
                
#                             result_string = cls.process_output(config, meta['short_name'], opt_list,
#                                                                stdout_filename, stderr_filename)
#                             print(OUTPUT_DIVIDER_STRING)
#                             print("\nResult: " + result_string)
#                             print(OUTPUT_DIVIDER_STRING)
#                             if "FAILED" in result_string:
#                                 failed_tests.append(meta['short_name'])
#                                 
#                             
#                             out_file.write(result_string + "\n")
#                             out_file.flush()

            if num_tests_to_run == 1:
                print("Finished running %d test in %s." % (num_tests_to_run, test_group_name))
            else:
                print("Finished running %d tests in %s." % (num_tests_to_run, test_group_name))
#             print("\nNumber of failed tests: %d, failed tests: %s" %
#                   (len(failed_tests), ",".join(failed_tests)))
            print(OUTPUT_DIVIDER_STRING)

    @classmethod
    def get_spark_submit_cmd(cls, spark_settings, main_class, output_settings, meta_settings, opt_list):# , stdout_filename,stderr_filename
        spark_submit = "spark-submit"

        cmd = "%s %s --class %s %s %s %s %s" % (
            spark_submit, " ".join(spark_settings), main_class, cls.test_jar_path, 
            " ".join(output_settings), " ".join(meta_settings), " ".join(opt_list))
        return cmd

    @classmethod
    def process_output(cls, short_name, opt_list, stdout_filename, stderr_filename):
        with open(stdout_filename, "r") as stdout_file:
            output = stdout_file.read()
        results_token = "results: "
        result_string = ""
        if results_token not in output:
            result_string = "FAILED"
        else:
            result_line = filter(lambda x: results_token in x, output.split("\n"))[-1]
            result_json = result_line.replace(results_token, "")
            try:
                result_dict = json.loads(result_json)
            except:
                print("Failed to parse JSON:\n", result_json)
                raise

            num_results = len(result_dict['results'])
            err_msg = ("Expecting at least %s results "
                       "but only found %s" % (config.IGNORED_TRIALS + 1, num_results))
            assert num_results > config.IGNORED_TRIALS, err_msg

            # 2 modes: prediction problems (4 metrics) and others (time only)
            if 'trainingTime' in result_dict['results'][0]:
                # prediction problem
                trainingTimes = [r['trainingTime'] for r in result_dict['results']]
                testTimes = [r['testTime'] for r in result_dict['results']]
                trainingMetrics = [r['trainingMetric'] for r in result_dict['results']]
                testMetrics = [r['testMetric'] for r in result_dict['results']]
                trainingTimes = trainingTimes[config.IGNORED_TRIALS:]
                testTimes = testTimes[config.IGNORED_TRIALS:]
                trainingMetrics = trainingMetrics[config.IGNORED_TRIALS:]
                testMetrics = testMetrics[config.IGNORED_TRIALS:]
                result_string += "Training time: %s, %.3f, %s, %s, %s\n" % \
                                 stats_for_results(trainingTimes)
                result_string += "Test time: %s, %.3f, %s, %s, %s\n" % \
                                 stats_for_results(testTimes)
                result_string += "Training Set Metric: %s, %.3f, %s, %s, %s\n" % \
                                 stats_for_results(trainingMetrics)
                result_string += "Test Set Metric: %s, %.3f, %s, %s, %s" % \
                                 stats_for_results(testMetrics)
            else:
                # non-prediction problem
                times = [r['time'] for r in result_dict['results']]
                times = times[config.IGNORED_TRIALS:]
                result_string += "Time: %s, %.3f, %s, %s, %s\n" % \
                                 stats_for_results(times)

        result_string = "%s, %s\n%s" % (short_name, " ".join(opt_list), result_string)

        sys.stdout.flush()
        return result_string

