package alchemist

import alchemist.io._

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
// spark-sql
import org.apache.spark.sql.SparkSession
// spark-mllib
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import scala.math.max


class MatrixHandle(val id: Int) extends Serializable { }

class AlchemistContext(client: DriverClient) extends Serializable {
  
  val workerIds: Array[WorkerId] = client.workerIds
  val workerInfo: Array[WorkerInfo] = client.workerInfo

  def connectWorker(worker: WorkerId): WorkerClient = workerInfo(worker.id).newClient
}

object Alchemist {
  System.err.println("Launching Alchemist")

  val driver = new Driver()
  val client = driver.client
  
  var sc: SparkContext = _
  
  // Instances of `Alchemist` are not serializable, but `.context`
  // has everything needed for RDD operations and is serializable.
  val context = new AlchemistContext(client)
  
  val libraries = collection.mutable.Map[String, String]()

  def registerLibrary(libraryInfo: (String, String)) {
    libraries.update(libraryInfo._1, libraryInfo._2)
    client.loadLibrary(libraryInfo._1, libraryInfo._2)
  }
  
  def listLibraries(): Unit = libraries foreach (x => println (x._1 + "-->" + x._2))
  
  def create(_sc: SparkContext) {
    sc = _sc
  }
  
  def run(libraryName: String, funName: String, inputParams: Parameters): Parameters = {
    client.runCommand(libraryName, funName, inputParams)
  }
  
  def getMatrixHandle(mat: IndexedRowMatrix): MatrixHandle = {
    val workerIds = context.workerIds
    // rowWorkerAssignments is an array of WorkerIds whose ith entry is the world rank of the alchemist worker
    // that will take the ith row (ranging from 0 to numworkers-1). Note 0 is an executor, not the driver
    try {
      val (handle, rowWorkerAssignments) = client.sendNewMatrix(mat.numRows, mat.numCols)
      
      mat.rows.mapPartitionsWithIndex { (idx, part) =>
        val rows = part.toArray
        val relevantWorkers = rows.map(row => rowWorkerAssignments(row.index.toInt).id).distinct.map(id => new WorkerId(id))
        val maxWorkerId = relevantWorkers.map(node => node.id).max
        var nodeClients = Array.fill(maxWorkerId+1)(None: Option[WorkerClient])
        System.err.println(s"Connecting to ${relevantWorkers.length} workers")
        relevantWorkers.foreach(node => nodeClients(node.id) = Some(context.connectWorker(node)))
        System.err.println(s"Successfully connected to all workers")
  
        // TODO: randomize the order the rows are sent in to avoid queuing issues?
        var count = 0
        rows.foreach{ row =>
          count += 1
  //        System.err.println(s"Sending row ${row.index.toInt}, ${count} of ${rows.length}")
          nodeClients(rowWorkerAssignments(row.index.toInt).id).get.
            newMatrixAddRow(handle, row.index, row.vector.toArray)
        }
        System.err.println("Finished sending rows")
        nodeClients.foreach(client => 
            if (client.isDefined) {
              client.get.newMatrixPartitionComplete(handle)
              client.get.close()
            })
        Iterator.single(true)
      }.count
      
      client.sendNewMatrixDone()
      
      handle
    }
    catch {
      case protocol: ProtocolException => System.err.println("Blech!")
      stop()
      new MatrixHandle(0)
    }
  }
  
  // Caches result by default, because may not want to recreate (e.g. if delete referenced matrix on Alchemist side to save memory)
  def getIndexedRowMatrix(handle: MatrixHandle): IndexedRowMatrix = {
    val (numRows, numCols) = getDimensions(handle)
    // TODO:
    // should map the rows back to the executors using locality information if possible
    // otherwise shuffle the rows on the MPI side before sending them back to SPARK
    val numPartitions = max(sc.defaultParallelism, client.workerCount)
    val sacrificialRDD = sc.parallelize(0 until numRows.toInt, numPartitions)
    val layout: Array[WorkerId] = (0 until sacrificialRDD.partitions.size).map(x => new WorkerId(x % client.workerCount)).toArray
    val fullLayout: Array[WorkerId] = (layout zip sacrificialRDD.mapPartitions(iter => Iterator.single(iter.size), true)
                                          .collect())
                                          .flatMap{ case (workerid, partitionSize) => Array.fill(partitionSize)(workerid) }

    client.getIndexedRowMatrix(handle, fullLayout)
    val rows = sacrificialRDD.mapPartitionsWithIndex( (idx, rowindices) => {
      val worker = context.connectWorker(layout(idx))
      val result = rowindices.toList.map { rowIndex =>
        new IndexedRow(rowIndex, worker.getIndexedRowMatrixRow(handle, rowIndex, numCols))
      }.iterator
      worker.close()
      result
    }, preservesPartitioning = true)
    val result = new IndexedRowMatrix(rows, numRows, numCols)
    result.rows.cache()
    result.rows.count
    result
  }
  
  def readHDF5(fname: String, varname: String): MatrixHandle = client.readHDF5(fname, varname)
  
  def getDimensions(handle: MatrixHandle): Tuple2[Long, Int] = client.getMatrixDimensions(handle)

  def transpose(mat: IndexedRowMatrix): IndexedRowMatrix = {
    getIndexedRowMatrix(client.getTranspose(getMatrixHandle(mat)))
  }
  
  def matrixMultiply(matA: IndexedRowMatrix, matB: IndexedRowMatrix): IndexedRowMatrix = {
    getIndexedRowMatrix(client.matrixMultiply(getMatrixHandle(matA), getMatrixHandle(matB)))
  }
    
  def stop() = driver.stop()
}
