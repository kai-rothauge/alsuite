package altest.nla

import org.json4s._
// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
// spark-sql
import org.apache.spark.sql.SparkSession
// spark-mllib
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.random.{RandomRDDs}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.SingularValueDecomposition
//breeze
import breeze.linalg.{DenseVector => BDV, max, min, DenseMatrix => BDM, norm, diag, svd}
import breeze.numerics._
// others
import alchemist.Alchemist
import alchemist.util.ConsolePrinter
import altest.util.{DataGenerator, DataLoader}
import allib.AlLib
import allib.nla.{TruncatedSVD => alTruncatedSVD}
import altest.PerfTest
import scala.math
import java.io._

/** Parent class for linear algebra tests which run on a large dataset.
  * Generated this way so that SVD / PCA can be added easily
  */
abstract class LinearAlgebraTest(spark: SparkSession, cp: ConsolePrinter) extends PerfTest(cp) {

  var trainRDD:     RowMatrix = _
  var testRDD:      RowMatrix = _
  var labelVecRDD:  RDD[(Int, Vector)] = _
  var labelVecRDD1: RDD[(Int, Vector)] = _
  
  var sparkTime: Double = _
  var alTimes: Array[Double] = _
  
  var startTime: Double = _
  
  val sc = spark.sparkContext
  sc.setLogLevel("ERROR")

  override def run(): Unit = {
//    val rank = intOptionValue(RANK)

    runTest
    
//    val start = System.currentTimeMillis()
//    runTest(rdd, rank)
//    val end = System.currentTimeMillis()
//    val time = (end - start).toDouble / 1000.0
//
//    Map("time" -> time)
//    
//    val jarr2 = JArray(List(JString("foo"),JInt(42)))
//    jarr2
  }
}

class SVDTest(spark: SparkSession, cp: ConsolePrinter) extends LinearAlgebraTest(spark, cp) {
  
  var numRows: Int = _
  var numCols: Int = _
  var rank: Int = _
  
  var NUM_ROWS = ("num-rows", "Number of rows of the matrix",      "Int", true)
  var NUM_COLS = ("num-cols", "Number of columns of the matrix",   "Int", true)
  val RANK     = ("rank",     "Number of leading singular values", "Int", true)
  
  parser.addOptionsToParser(NUM_ROWS, NUM_COLS, RANK)
  
  override def loadTestSettings(): Unit = {
    if (generateData) {
      numRows = parser.getOptionValue[Int](NUM_ROWS)
      numCols = parser.getOptionValue[Int](NUM_COLS)
    }
    rank = parser.getOptionValue[Int](RANK)
  }
  
  override def outputTestSettings(): Unit = {
    if (generateData) {
      cp.println("%s: %4d".format(NUM_ROWS._2, numRows))
      cp.println("%s: %4d".format(NUM_COLS._2, numCols))
    }
    else {
      cp.println("%s: %s".format(DATA_FILE._2, dataFile))
    }
    cp.println("%s: %4d".format(RANK._2, rank))
  }
  
  override def outputTestConclusion(): Unit = {
    
    printTimes()
  }
  
  override def createInputData(seed: Long) = {

//    val numExamples: Long = parser.getOptionValue[Long](NUM_EXAMPLES)
//    val numFeatures: Int = parser.getOptionValue[Int](NUM_FEATURES)
//    val numCenters: Int = parser.getOptionValue(NUM_CENTERS)
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
  
  override def testSpark(): Unit = {

    // Compute the Squared Frobenius Norm
    val sqFroNorm: Double = labelVecRDD.map(pair => Vectors.norm(pair._2, 2))
                                    .map(norm => norm * norm)
                                    .reduce((a, b) => a + b)
    
    // Spark Build-in Truncated SVD
    startTime = System.nanoTime()
    val mat: RowMatrix = new RowMatrix(labelVecRDD.map(pair => pair._2))
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(rank, computeU=false)
    val v: Matrix = svd.V
    cp.println("Time cost of Spark %s test:                               %6.4fs".format(longName, (System.nanoTime() - startTime)*1.0E-9))
    
    // Compute Approximation Error
    val vBroadcast = sc.broadcast(v)
    val err: Double = labelVecRDD
            .map(pair => (pair._2, vBroadcast.value.transpose.multiply(pair._2)))
            .map(pair => (pair._1, Vectors.dense(vBroadcast.value.multiply(pair._2).toArray)))
            .map(pair => Vectors.sqdist(pair._1, pair._2))
            .reduce((a, b) => a + b)
    val relativeError = err / sqFroNorm
    cp.println("Squared Frobenius error of rank %d SVD is %6.4f".format(rank, err))
    cp.println("Squared Frobenius norm of A is %6.4f".format(sqFroNorm))
    cp.println("Relative Error is %6.4f".format(relativeError))
  }
      
  override def testAlchemist(): Unit = {
    
    // Compute the Squared Frobenius Norm
    val sqFroNorm: Double = labelVecRDD1.map(pair => Vectors.norm(pair._2, 2))
                                    .map(norm => norm * norm)
                                    .reduce((a, b) => a + b)
         
    // Convert Data to Indexed Vectors and Labels
    startTime = System.nanoTime()
    val (sortedLabels, indexedMat) = splitLabelVec(labelVecRDD1)
    cp.println("Time cost of creating indexed vectors and labels:         %6.4fs".format((System.nanoTime() - startTime)*1.0E-9))
    
//    // Convert Spark IndexedRowMatrix to Alchemist AlMatrix
//    t1 = System.nanoTime()
//    val alMat = AlMatrix(al, indexedMat)
//    cp.println("Time cost of converting Spark matrix to Alchemist matrix: %6.4fs".format((System.nanoTime() - t1)*1.0E-9))
    
//    // Alchemist Truncated SVD
    val (alU, alS, alV, alTimes) = alTruncatedSVD.compute(indexedMat, rank)

    // Alchemist Matrix to Local Matrix
    startTime = System.nanoTime()
    val arrV: Array[Array[Double]] =  alV.rows.map(row => row.vector.toArray).collect
    val d = arrV.size
    val matV: Matrix = Matrices.dense(rank, d, arrV.flatten)
    cp.println("Time cost of Alchemist matrix to local matrix:            %6.4fs".format((System.nanoTime() - startTime)*1.0E-9))
//     cp.println("Number of rows of V: " + matV.numRows.toString)
//     cp.println("Number of columns of V: " + matV.numCols.toString)
//     cp.println(" ")
    
    // Compute approximation error
    val vBroadcast = sc.broadcast(matV)
    val err: Double = labelVecRDD1
            .map(pair => (pair._2, vBroadcast.value.multiply(pair._2)))
            .map(pair => (pair._1, Vectors.dense(vBroadcast.value.transpose.multiply(pair._2).toArray)))
            .map(pair => Vectors.sqdist(pair._1, pair._2))
            .reduce((a, b) => a + b)
    val relativeError = err / sqFroNorm
    cp.println("Squared Frobenius error of rank %d SVD is %6.4f".format(rank, err))
    cp.println("Squared Frobenius norm of A is %6.4f".format(sqFroNorm))
    cp.println("Relative Error is %6.4f".format(relativeError))
  }
  
  def printTimes(): Unit = {
    cp.println("Spark time cost:")
    cp.println("    Spark matrix multiplication:                        %6.4fs".format(sparkTime*1.0E-9))
    cp.println("  ")
    cp.println("Alchemist time costs:")
    cp.println("    Converting Spark matrices to Alchemist matrices:    %6.4fs".format(alTimes(0)*1.0E-9))
    cp.println("    Alchemist matrix multiplication:                    %6.4fs".format(alTimes(1)*1.0E-9))
    cp.println("    Converting Alchemist matrix to Spark matrix:        %6.4fs".format(alTimes(2)*1.0E-9))
    cp.println("    -----------------------------------------------------------------")
    cp.println("    Total:                                              %6.4fs".format((alTimes(0)+alTimes(1)+alTimes(2))*1.0E-9))
  }
  
  def splitLabelVec(labelVecRDD: RDD[(Int, Vector)]): (Array[Int], IndexedRowMatrix) = {
    // Convert Data to Indexed Vectors and Labels
    val indexlabelVecRDD = labelVecRDD.zipWithIndex.persist()
    val sortedLabels = indexlabelVecRDD.map(pair => (pair._2, pair._1._1))
                                    .collect
                                    .sortWith(_._1 < _._1)
                                    .map(pair => pair._2)
                                    
    //sortedLabels.take(20).foreach(println)
    
    val indexRows = indexlabelVecRDD.map(pair => new IndexedRow(pair._2, new DenseVector(pair._1._2.toArray)))
    //print(indexRows.take(10).map(pair => (pair._1, pair._2.mkString(", "))).mkString(";\n"))
    val indexedMat = new IndexedRowMatrix(indexRows)
    
    (sortedLabels, indexedMat)
  }
}

class MatrixMultiplyTest(spark: SparkSession, cp: ConsolePrinter) extends LinearAlgebraTest(spark, cp) {
  
  var m: Int         = _
  var k: Int         = _
  var n: Int         = _
  var scaleA: Double = _
  var scaleB: Double = _

  var M = ("M", "Number of rows of matrix A",                     "Int", false)
  var K = ("K", "Number of columns of matrix A/rows of matrix B", "Int", false)
  var N = ("N", "Number of columns of matrix B",                  "Int", false)
  var SCALE_A = ("scale-A", "Scaling for matrix A",               "Double", false)
  var SCALE_B = ("scale-B", "Scaling for matrix B",               "Double", false)
  
  parser.addOptionsToParser(M, K, N, SCALE_A, SCALE_B)
  
  var matA: IndexedRowMatrix = _
  var matB: IndexedRowMatrix = _

  var sparkRes: IndexedRowMatrix = _
  var alRes: IndexedRowMatrix = _
  
  override def loadTestSettings(): Unit = {
    if (generateData) {
      m      = parser.getOptionValue[Int](M)
      k      = parser.getOptionValue[Int](K)
      n      = parser.getOptionValue[Int](N)
      scaleA = parser.getOptionValue[Double](SCALE_A)
      scaleB = parser.getOptionValue[Double](SCALE_B)
    }
  }
  
  override def outputTestSettings(): Unit = {
    if (generateData) {
      cp.println("%s: %6d".format(M._2, m))
      cp.println("%s: %6d".format(K._2, k))
      cp.println("%s: %6d".format(N._2, n))
      cp.println("%s: %4.2f".format(SCALE_A._2, scaleA))
      cp.println("%s: %4.2f".format(SCALE_B._2, scaleB))
    }
    else {
      cp.println("%s: %s".format(DATA_FILE._2, dataFile))
    }
  }
  
  override def outputTestConclusion(): Unit = {
    
    val sparkResLocalMat = toLocalMatrix(sparkRes)
    val alResLocalMat    = toLocalMatrix(alRes)
    val diff             = norm(alResLocalMat.toDenseVector - sparkResLocalMat.toDenseVector)
    cp.println("The Frobenius norm difference between Spark and Alchemist's results is %6.4f".format(diff))
    cp.println("  ")
    printTimes()
  }

  override def createInputData(seed: Long) = {
    
    matA = deterministicMatrix(sc, m, k, scaleA)
    matA.rows.cache
    matB = deterministicMatrix(sc, k, n, scaleB)
    matB.rows.cache
  }
  
  override def loadInputData() = {
    
    dataFile = parser.getOptionValue[String](DATA_FILE)
    
    // Nothing here yet
  }
  
  override def testSpark(): Unit = {                  // Spark matrix multiply
    
    startTime = System.nanoTime()
    sparkRes = matA.toBlockMatrix(matA.numRows.toInt, matA.numCols.toInt)
                            .multiply(matB.toBlockMatrix(matB.numRows.toInt, matB.numCols.toInt))
                            .toIndexedRowMatrix
    sparkTime = System.nanoTime() - startTime
  }
      
  override def testAlchemist(): Unit = {              // Alchemist matrix multiply
    
    val (alRes0, alTimes0) = Alchemist.matrixMultiply(matA, matB)
    alRes = alRes0
    alTimes = alTimes0
  }
  
  def printTimes(): Unit = {
    cp.println("Spark time cost:")
    cp.println("    Spark matrix multiplication:                        %6.4fs".format(sparkTime*1.0E-9))
    cp.println("  ")
    cp.println("Alchemist time costs:")
    cp.println("    Converting Spark matrices to Alchemist matrices:    %6.4fs".format(alTimes(0)*1.0E-9))
    cp.println("    Alchemist matrix multiplication:                    %6.4fs".format(alTimes(1)*1.0E-9))
    cp.println("    Converting Alchemist matrix to Spark matrix:        %6.4fs".format(alTimes(2)*1.0E-9))
    cp.println("    -----------------------------------------------------------------")
    cp.println("    Total:                                              %6.4fs".format((alTimes(0)+alTimes(1)+alTimes(2))*1.0E-9))
  }
  
  implicit def arrayOfIntsToLocalMatrix(arr: Array[Int]) : BDM[Double] = {
    new BDM(arr.length, 1, arr.toList.map(_.toDouble).toArray)
  }

  implicit def indexedRowMatrixToLocalMatrix(mat: IndexedRowMatrix) : BDM[Double] = {
    toLocalMatrix(mat)
  }

  def toLocalMatrix(mat: IndexedRowMatrix) : BDM[Double] = {
    val numRows = mat.numRows.toInt
    val numCols = mat.numCols.toInt
    val res = BDM.zeros[Double](numRows, numCols)
    mat.rows.collect.foreach{ indexedRow => res(indexedRow.index.toInt, ::) := BDV(indexedRow.vector.toArray).t }
    res
  }

  def displayBDM(mat : BDM[Double], truncationLevel : Double = 1e-10) = {
    cp.println("%d x %d".format(mat.rows, mat.cols))
    (0 until mat.rows).foreach{ i => println(mat(i, ::).t.toArray.map( x => if (x <= truncationLevel) 0 else x).mkString(" ")) }
  }

  def deterministicMatrix(sc: SparkContext, numRows: Int, numCols: Int, scale: Double): IndexedRowMatrix = {
    val mat = BDM.zeros[Double](numRows, numCols) 
    (0 until min(numRows, numCols)).foreach { i : Int => mat(i, i) = scale * (i + 1)}
    val rows = sc.parallelize( (0 until numRows).map( i => mat(i, ::).t.toArray )).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, new DenseVector(x._1))))
  }

  def randomMatrix(sc: SparkContext, numRows: Int, numCols: Int): IndexedRowMatrix = {
    val rows = RandomRDDs.normalVectorRDD(sc, numRows, numCols, 128).zipWithIndex
    new IndexedRowMatrix(rows.map(x => new IndexedRow(x._2, x._1)))
  }
}