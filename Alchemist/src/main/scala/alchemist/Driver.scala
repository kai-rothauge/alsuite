package alchemist

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import scala.collection.JavaConverters._
import scala.util.Random
import java.io.{
    PrintWriter, FileOutputStream,
    InputStream, OutputStream,
    DataInputStream => JDataInputStream,
    DataOutputStream => JDataOutputStream
}
import java.nio.{
    DoubleBuffer, ByteBuffer
}
import scala.io.Source
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import scala.compat.Platform.EOL

import alchemist._


class Driver {
  val listenSock = new java.net.ServerSocket(0);
  
  val driverProc: Process = {
    val pb = {
      if (System.getenv("NERSC_HOST") != null) {
        val sparkDriverNode = s"${System.getenv("SPARK_MASTER_NODE")}"
        val hostfilePath = s"${System.getenv("SPARK_WORKER_DIR")}/hosts.alchemist"
        val sockPath = s"${System.getenv("SPARK_WORKER_DIR")}/connection.info"

        val pw = new PrintWriter(new FileOutputStream(sockPath, false))
        pw.write(s"${sparkDriverNode},${listenSock.getLocalPort().toString()}")
        pw.close
        
        // dummy process
        new ProcessBuilder("true")
      }
      else {
        new ProcessBuilder("mpirun", "-q", "-np", "3", sys.env("ALCHEMIST_EXE"), "localhost", 
            listenSock.getLocalPort().toString())
      }
    }
//    pb.redirectError(ProcessBuilder.Redirect.INHERIT).start
    pb.redirectError(ProcessBuilder.Redirect.INHERIT).redirectOutput(ProcessBuilder.Redirect.INHERIT).start
  }
    
  val driverSock = listenSock.accept()
  System.err.println(s"Alchemist.Driver: Accepting connection from Alchemist driver on socket")
  val client = new DriverClient(driverSock.getInputStream, driverSock.getOutputStream)
                    .handshake()
                    .getWorkerInfo()

  def stop(): Unit = {
    client.shutdown
    driverProc.waitFor
  }
}