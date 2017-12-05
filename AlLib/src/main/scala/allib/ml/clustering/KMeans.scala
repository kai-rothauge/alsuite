package allib.ml.clustering

import allib._
import alchemist.{Alchemist, Parameters}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
 
object KMeans {
  def train(mat: IndexedRowMatrix, k: Int = 2, maxIterations: Int = 20, epsilon: Double = 1e-4,
      initMode: String = "k-means||", initSteps: Int = 2, seed: Long = 0): (IndexedRowMatrix, IndexedRowMatrix, Int) = {

    val matHandle = Alchemist.getMatrixHandle(mat)
    
    val inParameters = Parameters()
    
    inParameters.addParameter("data", "matrix handle", matHandle.toString)
    inParameters.addParameter("num_centers", "int", k.toString)
    inParameters.addParameter("max_iterations", "int", maxIterations.toString)
    inParameters.addParameter("epsilon", "double", epsilon.toString)
    inParameters.addParameter("init_mode", "string", initMode)
    inParameters.addParameter("init_steps", "int", initSteps.toString)
    inParameters.addParameter("seed", "long", seed.toString)
    
    println("Calling 'Alchemist.run'")
    val outParameters = Alchemist.run(AlLib.getName(), "kmeans", inParameters)

    val clusters = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("clusters"))
    val assignments = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("assignments"))
    val numIters = outParameters.getInt("numIters")
    
    (clusters, assignments, numIters)
  }
}
