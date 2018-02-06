package altest

import scala.collection.JavaConverters._
import joptsimple.{OptionSet, OptionParser}
import java.io.PrintWriter
import org.json4s._
import alchemist.util.ConsolePrinter
import altest.util.ArgParser

abstract class PerfTest(cp: ConsolePrinter) {
  
  var shortName: String = _
  var longName: String = _
  var generateData: Boolean = true
  var dataFile: String = _
  
  var sparkResultsWriter: PrintWriter = _
  var alchemistResultsWriter: PrintWriter = _
  
  var rand = new java.util.Random(0)

  val SHORT_NAME       = ("short-name",       "short name of algorithm to be tested",  "String", true)
  val LONG_NAME        = ("long-name",        "long name of algorithm to be tested",   "String", false)
  val GENERATE_DATA    = ("generate-data",    "Generate data or load it from file",    "String", true)
  val NUM_TRIALS       = ("num-trials",       "number of trials to run",               "Int",    false)
  val INTER_TRIAL_WAIT = ("inter-trial-wait", "seconds to sleep between trials",       "Int",    false)
  val NUM_PARTITIONS   = ("num-partitions",   "number of input partitions",            "Int",    false)
  val RANDOM_SEED      = ("random-seed",      "Seed for random number generator",      "Long",   false)
  val DATA_FILE        = ("data-file",        "File containing data",                  "String", false)
  
  val parser = new ArgParser()
  parser.addOptionsToParser(SHORT_NAME, LONG_NAME, GENERATE_DATA, NUM_TRIALS, INTER_TRIAL_WAIT, NUM_PARTITIONS, RANDOM_SEED, DATA_FILE)

  /** Initialize internal state based on arguments */
  def initialize(args: Array[String]) {
//    cp.println("PerfTest.initialize start") 
    parser.parse(args.slice(0, args.length))
    
    shortName = getShortName
    longName  = getLongName
    generateData = (parser.getOptionValue[String](GENERATE_DATA) == "True")
    
    loadTestSettings
    
//    cp.println("PerfTest.initialize end")
  }
  
  def setOutputWriters(sparkWriter: PrintWriter, alchemistWriter: PrintWriter) {
    sparkResultsWriter     = sparkWriter
    alchemistResultsWriter = alchemistWriter
  }
  
  def getShortName: String = {
    parser.getOptionValue[String](SHORT_NAME).replace("0x20", " ")
  }
  
  def getLongName: String = {
    parser.getOptionValue[String](LONG_NAME).replace("0x20", " ")
  }
  
  def getRandomSeed: Long = {
    parser.getOptionValue[Long](RANDOM_SEED)
  }

  def getNumTrials: Int = {
    parser.getOptionValue[Int](NUM_TRIALS)
  }

  def getWait: Int = {
    parser.getOptionValue[Int](INTER_TRIAL_WAIT) * 1000
  }

  def prepareInputData {
    if (!generateData) {
      loadInputData()
    } else {
      // Generate a new dataset for each test
      rand = new java.util.Random(getRandomSeed)
      createInputData(rand.nextLong())
    }
  }
  
  def runTest {
    
    cp.println(" ")
    cp.println("Settings:")
    cp.println("---------\n")
    cp.tab
    outputTestSettings
    cp.untab
    cp.println(" ")
      
    // Test Spark
    cp.println("Testing Spark:")
    cp.println("--------------\n")
    cp.tab
    testSpark
    cp.untab
    cp.println(" ")
    
    // Test Alchemist
    cp.println("Testing Alchemist:")
    cp.println("------------------\n")
    cp.tab
    testAlchemist
    cp.untab
    cp.println(" ")
    
    outputTestConclusion
    cp.println(" ")
  }
  
  def createInputData(seed: Long): Unit
  def loadInputData(): Unit

//  /**
//   * Runs the test and returns a JSON object that captures performance metrics, such as time taken,
//   * and values of any parameters.
//   *
//   * The rendered JSON will look like this (except it will be minified):
//   *
//   *    {
//   *       "options": {
//   *         "num-partitions": "10",
//   *         "unique-values": "10",
//   *         ...
//   *       },
//   *       "results": [
//   *         {
//   *           "trainingTime": 0.211,
//   *           "trainingMetric": 98.1,
//   *           ...
//   *         },
//   *         ...
//   *       ]
//   *     }
//   *
//   * @return metrics from run (e.g. ("time" -> time)
//   *  */
  def run(): Unit
  
  def loadTestSettings(): Unit
  def outputTestSettings(): Unit
  def outputTestConclusion(): Unit
  
  def testSpark(): Unit
  def testAlchemist(): Unit  
  
  def getOptions: Map[String, String] = parser.getOptions
}

//  val parser = new OptionParser()
//  var optionSet: OptionSet = _
//  var shortName: String = _
//  var longName: String = _
//  var inFile: String = _

//  var intOptions: Seq[(String, String)] = Seq(NUM_TRIALS, INTER_TRIAL_WAIT, NUM_PARTITIONS,
//    RANDOM_SEED)
//
//  var doubleOptions: Seq[(String, String)] = Seq()
//  var longOptions: Seq[(String, String)] = Seq()
//
//  var stringOptions: Seq[(String, String)] = Seq()
//  var booleanOptions: Seq[(String, String)] = Seq()
//
//  def addOptionsToParser() {
//    // add all the options to parser
//    stringOptions.map{case (opt, desc) =>
//      parser.accepts(opt, desc).withRequiredArg().ofType(classOf[String]).required()
//    }
//    booleanOptions.map{case (opt, desc) =>
//      parser.accepts(opt, desc)
//    }
//    intOptions.map{case (opt, desc) =>
//      parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Int]).required()
//    }
//    doubleOptions.map{case (opt, desc) =>
//      parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Double]).required()
//    }
//    longOptions.map{case (opt, desc) =>
//      parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Long]).required()
//    }
//  }
//
//  def addOptionalOptionToParser[T](opt: String, desc: String, default: T, clazz: Class[T]): Unit = {
//    parser.accepts(opt, desc).withOptionalArg().ofType(clazz).defaultsTo(default)
//  }
//
//  def intOptionValue(option: (String, String)) =
//    optionSet.valueOf(option._1).asInstanceOf[Int]
//
//  def stringOptionValue(option: (String, String)) =
//    optionSet.valueOf(option._1).asInstanceOf[String]
//
//  def booleanOptionValue(option: (String, String)) =
//    optionSet.has(option._1)
//
//  def doubleOptionValue(option: (String, String)) =
//    optionSet.valueOf(option._1).asInstanceOf[Double]
//
//  def longOptionValue(option: (String, String)) =
//    optionSet.valueOf(option._1).asInstanceOf[Long]
//
//  def optionValue[T](option: String) =
//    optionSet.valueOf(option).asInstanceOf[T]