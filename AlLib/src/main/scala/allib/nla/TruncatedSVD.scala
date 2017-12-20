package allib.nla

import allib._
import alchemist.{Alchemist, Parameters}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
 
object TruncatedSVD {
  def compute(mat: IndexedRowMatrix, rank: Int = 10): (IndexedRowMatrix, IndexedRowMatrix, IndexedRowMatrix, Array[Double]) = {

    var t1 = System.nanoTime()
    val matHandle = Alchemist.getMatrixHandle(mat)
    val t2 = System.nanoTime() - t1
    
    val inParameters = Parameters()
    
    inParameters.addParameter("data", "matrix handle", matHandle.id.toString)
    inParameters.addParameter("rank", "int", rank.toString)
    
    println("Calling 'Alchemist.run'")
    t1 = System.nanoTime()
    val outParameters = Alchemist.run(AlLib.getName(), "truncated_svd", inParameters)
    val t3 = System.nanoTime() - t1

    t1 = System.nanoTime()
    val U = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("U"))
    val S = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("S"))
    val V = Alchemist.getIndexedRowMatrix(outParameters.getMatrixHandle("V"))
    val t4 = System.nanoTime() - t1
    
    (U, S, V, Array(t2, t3, t4))
  }
}
