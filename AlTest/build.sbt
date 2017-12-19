scalaVersion := "2.11.8"
name := "altest"
version := "0.1"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0" % "provided"
libraryDependencies += "net.sf.jopt-simple" % "jopt-simple" % "5.0.4"

test in assembly := {}

mainClass in assembly := Some("altest.AlTest")