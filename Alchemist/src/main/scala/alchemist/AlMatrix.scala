package alchemist

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import scala.math.max

class MatrixHandle(val id: Int) extends Serializable { }

class AlMatrix(val handle: MatrixHandle) {
  
  def transpose() : AlMatrix = {
    new AlMatrix(al, al.client.getTranspose(handle))
  }
  
  
}