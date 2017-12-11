package allib

import alchemist.util.ConsolePrinter
import alchemist._
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


object AlLib {
    
  val libraryName: String = "AlLib"
  var libraryPath: String = sys.env("ALLIB_DYLIB")
  
  def getName(): String = libraryName
  
  def getLibraryPath(): String = libraryPath
  
  def setLibraryPath(path: String): Unit = {
    libraryPath = path 
  }
  
  def register(): Unit = {
    Alchemist.registerLibrary(libraryName, libraryPath)
  }
  
  def getRegistrationInfo = (libraryName, libraryPath)
}




