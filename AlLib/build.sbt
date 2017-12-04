scalaVersion := "2.11.8"
name := "allib"
version := "0.1"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0" % "provided"

test in assembly := {}

