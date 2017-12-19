package altest

import scala.collection.JavaConverters._

import org.json4s.JsonAST._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import java.text.SimpleDateFormat
import java.util.{Date}
import java.io.{File, PrintWriter}

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
// spark-sql
import org.apache.spark.sql.SparkSession

import alchemist.Alchemist
import alchemist.util.ConsolePrinter
import allib.AlLib
import altest.util.ArgParser
import altest.ml.clustering._
import altest.nla._

object AlTest {
  
  var logWriter: PrintWriter = _
  var errWriter: PrintWriter = _
  var sparkResultsWriter: PrintWriter = _
  var alchemistResultsWriter: PrintWriter = _
  
  def main(args: Array[String]) {

    if (args.length < 1) {
      println("AlTest requires 1 or more args, you gave %s, exiting".format(args.length))
      System.exit(1)
    }

    val perfTestArgs = args.slice(0, args.length)
    
    val CONSOLE_COLOR = ("console-color", "Console color of Alchemist Tester output", "String", false)
    val LOG_LEVEL     = ("log-level",     "Level of detail included in output",       "String", false)
    val SHORT_NAME    = ("short-name",    "Short name of algorithm to be tested",     "String", true)
    val LONG_NAME     = ("long-name",     "Long name of algorithm to be tested",      "String", false)
    val OUT_DIR       = ("out-dir",       "Path of output directory",                 "String", true)
    
    val parser = new ArgParser()
    parser.addOptionsToParser(CONSOLE_COLOR, LOG_LEVEL, SHORT_NAME, LONG_NAME, OUT_DIR)
    parser.parse(perfTestArgs)
    
    val consoleColor = parser.getOptionValue[String](CONSOLE_COLOR).replace("0x20", " ")
    val logLevel     = {
      val temp = parser.getOptionValue[String](LOG_LEVEL).replace("0x20", " ")
      if (temp.isEmpty()) "none" else temp
    }
    val shortName    = parser.getOptionValue[String](SHORT_NAME).replace("0x20", " ")
    val longName     = {
      val temp = parser.getOptionValue[String](LONG_NAME).replace("0x20", " ")
      if (temp.isEmpty()) shortName else temp
    }
    var outDir       = parser.getOptionValue[String](OUT_DIR).replace("0x20", " ")
  
    val cp = ConsolePrinter(consoleColor)
    
    val outputFilePrefix: String = {
      val dNow: Date = new Date( );
      val ft: SimpleDateFormat = new SimpleDateFormat("yyyy.MM.dd_HH:mm:ss");
      
      val dir: File = new File(outDir + shortName + "_latest/")
      
      if (dir.exists()) delete(dir)
  
      if (dir.mkdir()) {
        cp.println("Output directory '%s' was created successfully".format(dir))
        outDir = dir.toString + "/"
      }
      else {
        cp.println("Failed trying to create the directory '%s'".format(dir))
        cp.printError("Failed trying to create the directory '%s'".format(dir))
      }
      cp.println("Output directory set to '%s'".format(outDir))
      cp.println(" ")
      
      outDir + shortName + "_" + ft.format(dNow)
    }

    logWriter              = createOutputWriter(outputFilePrefix + ".log")
    errWriter              = createOutputWriter(outputFilePrefix + ".err")
    sparkResultsWriter     = createOutputWriter(outputFilePrefix + "_results_spark.out")
    alchemistResultsWriter = createOutputWriter(outputFilePrefix + "_results_alchemist.out")
    
    cp.setLogWriter(logWriter)
    cp.setErrWriter(errWriter)
    
//    cp.println("============= BEGIN TEST RUNNER ==============\n")
    
    cp.println("========================================================")
    cp.println("Creating Spark session")
    cp.println("--------------------------------------------------------\n")
    cp.tab
    var t1 = System.nanoTime()
    val appSparkName: String = "Alchemist: " + longName + " Test"
    cp.println("appSparkName: " + appSparkName)
    val spark = SparkSession.builder().appName(appSparkName).getOrCreate()
    cp.println("Time cost of starting Spark session: %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    cp.println(" ")
    cp.println("spark.conf.getAll:")
    spark.conf.getAll.foreach(line => cp.println(line))
    cp.println(" ")
    cp.println("getExecutorMemoryStatus:")
    cp.println(spark.sparkContext.getExecutorMemoryStatus.toString())
    cp.untab
    cp.println("\n========================================================\n")
    
    // Launch Alchemist
    cp.println("========================================================")
    cp.println("Creating Alchemist session")
    cp.println("--------------------------------------------------------\n")
    cp.tab
    t1 = System.nanoTime()
    Alchemist.create(spark.sparkContext)
    cp.println("Time cost of starting Alchemist session: %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    cp.untab
    cp.println("\n========================================================\n")
  
    cp.println("========================================================")
    if (longName == shortName) cp.println("Running %s test".format(shortName))
    else cp.println("Running %s (%s) test".format(longName,shortName))
    cp.println("--------------------------------------------------------\n")
    cp.tab

//      cp.println("Building new %s test object",shortName)
    val test: PerfTest = shortName match {
      // clustering
      case "kmeans" => new KMeansTest(spark,cp)
      // regression
//      case "regression" => new LinearRegressionTest(spark,al)
      // linalg
      case "svd" => new SVDTest(spark,cp)
      case "matmult" => new MatrixMultiplyTest(spark,cp)
    }
//      cp.println("Done building")
//      cp.println("Initializing %s test object",shortName)
    test.initialize(perfTestArgs)
    test.setOutputWriters(sparkResultsWriter, alchemistResultsWriter)
  
    val numTrials = test.getNumTrials
    val interTrialWait = test.getWait
    cp.println("numTrials      = %d".format(numTrials))
    cp.println("interTrialWait = %dms".format(interTrialWait))

    var testOptions: JValue = test.getOptions
    (1 to numTrials).map { i =>
      cp.println(" ")
      cp.println("Preparing input data")
      cp.tab
      test.prepareInputData
      cp.untab
      cp.println("Done preparing input data")
      cp.println(" ")
      cp.println("Running trial %d of %d:".format(i,numTrials))
      cp.tab
      test.run
      cp.untab
      cp.println("Completed trial %d of %d".format(i,numTrials))
      System.gc()
      Thread.sleep(interTrialWait)
    }
    
    cp.untab
    cp.println("\n========================================================\n")
    
//      val results: Seq[JValue] = (1 to numTrials).map { i =>
//        cp.println("Creating input data")
//        test.createInputData(rand.nextLong())
//        cp.println("Done creating input data")
//        cp.println("Running trial %s of %s",i.toString,numTrials.toString)
//        val res: JValue = test.run()
//        cp.println("Done Running",shortName)
//        System.gc()
//        Thread.sleep(interTrialWait)
//        res
//      }
//      // Report the test results as a JSON object describing the test options, Spark
//      // configuration, Java system properties, as well as the per-test results.
//      // This extra information helps to ensure reproducibility and makes automatic analysis easier.
//      val json: JValue =
//        ("shortName" -> shortName) ~
//        ("options" -> testOptions) ~
//        ("sparkConf" -> spark.conf.getAll.toMap) ~
//        ("sparkVersion" -> spark.version) ~
//        ("systemProperties" -> System.getProperties.asScala.toMap) ~
//        ("results" -> results)
//      cp.println("results: " + compact(render(json)))

  
    Alchemist.stop()
    spark.stop()
    
//    cp.println("============== END TEST RUNNER ===============\n")
    
    closeOutputWriters
  }
  
  def delete(file: File): Unit = {
    if (file.isDirectory) 
      Option(file.listFiles).map(_.toList).getOrElse(Nil).foreach(delete(_))
    file.delete
  }
  
  def createOutputWriter(fileName: String): PrintWriter = new PrintWriter(new File(fileName))
  
  def closeOutputWriters: Unit = {
    logWriter.close()
    errWriter.close()
    sparkResultsWriter.close()
    alchemistResultsWriter.close()
  }
}
