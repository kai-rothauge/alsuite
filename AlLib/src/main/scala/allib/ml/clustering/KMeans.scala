package allib.ml.clustering

import allib._
import alchemist.{Alchemist, Parameters}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
 
object KMeans {
  def train(mat: IndexedRowMatrix, k: Int = 2, maxIterations: Int = 20, epsilon: Double = 1e-4,
      initMode: String = "k-means||", initSteps: Int = 2, seed: Long = 0): (IndexedRowMatrix, IndexedRowMatrix, Int, Array[Double]) = {

    
    var t0 = System.nanoTime()
    val matHandle = Alchemist.getMatrixHandle(mat)
    val t1 = System.nanoTime() - t0
    
    val inParameters = Parameters()
    
    inParameters.addParameter("data", "matrix handle", matHandle.id.toString)
    inParameters.addParameter("num_centers", "int", k.toString)
    inParameters.addParameter("max_iterations", "int", maxIterations.toString)
    inParameters.addParameter("epsilon", "double", epsilon.toString)
    inParameters.addParameter("init_mode", "string", initMode)
    inParameters.addParameter("init_steps", "int", initSteps.toString)
    inParameters.addParameter("seed", "long", seed.toString)
    
    println("Calling 'Alchemist.run'")
    t0 = System.nanoTime()
    val outParameters = Alchemist.run(AlLib.getName(), "kmeans", inParameters)
    val t2 = System.nanoTime() - t0

    t0 = System.nanoTime()
    val centers = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("centers"))
    val assignments = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("assignments"))
    val numIters = outParameters.getInt("num_iterations")
    val t3 = System.nanoTime() - t0
    
    (centers, assignments, numIters, Array(t1, t2, t3))
  }
}
