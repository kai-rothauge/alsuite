package alchemist

import scala.collection._


object Main extends App {
  println("Testing Alchemist:")
  Alchemist.sayHello()
  Alchemist.registerLibrary("libA", "/usr/kai/here")
  Alchemist.registerLibrary("libB", "/usr/kai/there")
  Alchemist.listLibraries()
  
  val inParameters = Parameters()
  inParameters.addParameter("k", "int", "47")
  inParameters.addParameter("A", "string", "blah")
  inParameters.addParameter("numIters", "float", "3.14")
  
  Alchemist.run("libA", "kmeans", inParameters)
}