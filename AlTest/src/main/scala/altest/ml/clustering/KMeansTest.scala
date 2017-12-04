package altest.ml.clustering

import org.json4s._
// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// spark-mllib
import org.apache.spark.mllib.clustering.{KMeans => sparkKMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
//breeze
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag, svd}
import breeze.numerics._
// others
import alchemist._
import alchemist.util.{ConsolePrinter => cp}
import allib.AlLib
import allib.ml.clustering.{KMeans => alKMeans}

//import altest.util.{DataGenerator, DataLoader}

import scala.math
import java.io._

abstract class ClusteringTest(spark: SparkSession, al: Alchemist) extends PerfTest {

  var numIterations: Int = _
  var k: Int = _
  var seed: Long = _
  var threshold: Double = _
  var numExamples: Long = _
  var numFeatures: Int = _
  
  val NUM_CENTERS =      ("num-centers",      "Number of centers for clustering tests", "Int",    true)
  val NUM_ITERATIONS =   ("num-iterations",   "Number of iterations for the algorithm", "Int",    false)
  val CHANGE_THRESHOLD = ("change-threshold", "Change threshold for cluster centers",   "Double", false)
  
  parser.addOptionsToParser(NUM_CENTERS, NUM_ITERATIONS, CHANGE_THRESHOLD)
  
  
  var NUM_EXAMPLES: (String, String, String, Boolean) = _
  var NUM_FEATURES: (String, String, String, Boolean) = _
  
  if (generateData) {
    NUM_EXAMPLES = ("num-examples", "Number of examples for clustering tests",                  "Long",   false)
    NUM_FEATURES = ("num-features", "Number of features for each example for clustering tests", "Int",    false)
    
    parser.addOptionsToParser(NUM_EXAMPLES, NUM_FEATURES)
  }

  var trainRDD:     RDD[Vector] = _
  var testRDD:      RDD[Vector] = _
  var labelVecRDD:  RDD[(Int, Vector)] = _
  var labelVecRDD1: RDD[(Int, Vector)] = _
  
  val sc = spark.sparkContext
  sc.setLogLevel("ERROR")
  
  override def createInputData(seed: Long) = {
    
    numExamples = parser.getOptionValue[Long](NUM_EXAMPLES)
    numFeatures = parser.getOptionValue[Int](NUM_FEATURES)
      
//    
//    val data = DataGenerator.generateKMeansVectors(sc, math.ceil(numExamples*1.25).toLong, numFeatures,
//      numCenters, numPartitions, seed)
//
//    val split = data.randomSplit(Array(0.8, 0.2), seed)
//
//    trainRDD = split(0).cache()
//    testRDD  = split(1)
//    
//    cp.println("Num Examples: " + rdd.count())
  }
  
  override def loadInputData() = {
    
    dataFile = parser.getOptionValue[String](DATA_FILE)
    
    //  Load and Parse Data
    val t1 = System.nanoTime()
    val df = spark.read.format("libsvm").load(dataFile)
    labelVecRDD = df.rdd
            .map(pair => (pair(0).toString.toFloat.toInt, Vectors.parse(pair(1).toString)))
            .persist()
    labelVecRDD1 = df.rdd
            .map(pair => (pair(0).toString.toFloat.toInt, Vectors.parse(pair(1).toString)))
            .persist()
    
    cp.println("Number of partitions:      %d".format(labelVecRDD.getNumPartitions))
    cp.println("Time cost of loading data: %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
  }

  override def run(): Unit = {
      
//    var start = System.currentTimeMillis()
    runTest
//      val model = runTest()
//      val trainingTime = (System.currentTimeMillis() - start).toDouble / 1000.0
//  
//      start = System.currentTimeMillis()
//      val testTime = (System.currentTimeMillis() - start).toDouble / 1000.0
//  
//      val testMetric = validate(model, testRdd)
//      Map("trainingTime" -> trainingTime, "testTime" -> testTime,
//        "trainingMetric" -> trainingMetric, "testMetric" -> testMetric)
    
  }
}

// K-Means Clustering
class KMeansTest(spark: SparkSession, al: Alchemist) extends ClusteringTest(spark, al) {
  
  al.registerLibrary(AlLib.getRegistrationInfo)

  override def loadTestSettings(): Unit = {
    numIterations = parser.getOptionValue[Int](NUM_ITERATIONS)
    k             = parser.getOptionValue[Int](NUM_CENTERS)
    threshold     = parser.getOptionValue[Double](CHANGE_THRESHOLD)
    
    rand          = new java.util.Random(getRandomSeed)
    seed          = rand.nextLong()
  }
  
  override def outputTestSettings(): Unit = {
    cp.println(NUM_CENTERS._2 + ": %4d".format(k))
    cp.println(NUM_ITERATIONS._2 + ": %4d".format(numIterations))
    cp.println(RANDOM_SEED._2 + ": %4d".format(seed))
    cp.println(CHANGE_THRESHOLD._2 + ": %6.4f".format(threshold))
    if (generateData) {
      cp.println(NUM_EXAMPLES._2 + ": %4d".format(numExamples))
      cp.println(NUM_FEATURES._2 + ": %4d".format(numFeatures))
    }
    else {
      cp.println(DATA_FILE._2 + ": %s".format(dataFile))
    }
  }
  
  override def outputTestConclusion(): Unit = {

  }
    
  override def testSpark(): Unit = {  
    // Spark Build-in K-Means
    // val labelStr: String = kmeansSpark(sc, labelVecRDD, k, maxiters)
    // def kmeansSpark(sc: SparkContext, label_vector_rdd: RDD[(Int, Vector)], k: Int, maxiter: Int): String = {
    // K-Means Clustering
    
    var t1 = System.nanoTime()
    val clusters = sparkKMeans.train(labelVecRDD.map(pair => pair._2), k, numIterations, "k-means||", seed)
    val broadcast_clusters = sc.broadcast(clusters)
    val labels: Array[String] = labelVecRDD
            .map(pair => (pair._1, broadcast_clusters.value.predict(pair._2)))
            .map(pair => pair._1.toString + " " + pair._2.toString)
            .collect()
    
    // Print Info
    cp.println("Time cost of Spark k-means clustering:                            %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    cp.println("Final objective value:                                            %6.4f".format(validate(clusters, labelVecRDD1)))
    
    val labelStr: String = (labels mkString " ").trim

    // Write (true_label, predicted_label) pairs to file outfile
    sparkResultsWriter.write(labelStr)
  }
      
  override def testAlchemist(): Unit = {  

    // Alchemist K-Means
    // val labelStr: String = kmeansAlchemist(al, labelVecRDD, k, maxiters) 
    
    // Convert Data to Indexed Vectors and Labels
    var t1 = System.nanoTime()
    val (sortedLabels, indexedMat) = splitLabelVec(labelVecRDD1)
    cp.println("Time cost of creating indexed vectors and labels:                 %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    
    // Convert Spark IndexedRowMatrix to Alchemist AlMatrix
    t1 = System.nanoTime()
    val alMatkMeans = AlMatrix(al, indexedMat)
    cp.println("Time cost of converting Spark matrix to Alchemist matrix:         %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    
    // K-Means Clustering by Alchemist
    t1 = System.nanoTime()
    val (centers, assignments, numIters) = alKMeans(alMatkMeans, k, numIterations, threshold)
    cp.println("Time cost of Alchemist k-means clustering:                        %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    
    // Collect the clustering results
    t1 = System.nanoTime()
    val indexPredLabels = alAssignments.rows.map(row => (row.index, row.vector.toArray(0).toInt)).collect
    
    cp.println("Time cost of sending alchemist cluster assignments back to local: %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    
    val sortedPredLabels = indexPredLabels.sortWith(_._1 < _._1)
                                    .map(pair => pair._2)
    val labels = (sortedLabels zip sortedPredLabels).map(pair => pair._1.toString + " " + pair._2.toString)
    val labelStr: String = (labels mkString " ").trim
    
    // Write (true_label, predicted_label) pairs to file outfile
    alchemistResultsWriter.write(labelStr)
  }
  
  def splitLabelVec(labelVecRDD: RDD[(Int, Vector)]): (Array[Int], IndexedRowMatrix) = {
    // Convert Data to Indexed Vectors and Labels
    val indexlabelVecRDD = labelVecRDD.zipWithIndex.persist()
    val sortedLabels = indexlabelVecRDD.map(pair => (pair._2, pair._1._1))
                                    .collect
                                    .sortWith(_._1 < _._1)
                                    .map(pair => pair._2)
                                    
    //sortedLabels.take(20).foreach(cp.println)
    
    val indexRows = indexlabelVecRDD.map(pair => new IndexedRow(pair._2, new DenseVector(pair._1._2.toArray)))
    //print(indexRows.take(10).map(pair => (pair._1, pair._2.mkString(", "))).mkString(";\n"))
    val indexedMat = new IndexedRowMatrix(indexRows)
        
    (sortedLabels, indexedMat)
  }
  
  def validate(model: KMeansModel, rdd: RDD[(Int, Vector)]): Double = {
    val numExamples = rdd.cache().count()

    val error = model.computeCost(rdd.map(pair => pair._2))

    math.sqrt(error/numExamples)
  }
}
