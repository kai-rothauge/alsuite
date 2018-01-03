package allib.nla

import allib._
import alchemist.{Alchemist, Parameters}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
 
object TruncatedSVD {
  def compute(mat: IndexedRowMatrix, rank: Int = 10): (IndexedRowMatrix, IndexedRowMatrix, IndexedRowMatrix, Array[Double]) = {

    var t0 = System.nanoTime()
    val matHandle = Alchemist.getMatrixHandle(mat)
    val t1 = System.nanoTime() - t0
    
    val inParameters = Parameters()
    
    inParameters.addParameter("data", "matrix handle", matHandle.id.toString)
    inParameters.addParameter("rank", "int", rank.toString)
    
    println("Calling 'Alchemist.run'")
    t0 = System.nanoTime()
    val outParameters = Alchemist.run(AlLib.getName(), "truncated_svd", inParameters)
    val t2 = System.nanoTime() - t0

    t0 = System.nanoTime()
    val U = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("U"))
    val S = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("S"))
    val V = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("V"))
    val t3 = System.nanoTime() - t0
    
    (U, S, V, Array(t1, t2, t3))
  }
}
