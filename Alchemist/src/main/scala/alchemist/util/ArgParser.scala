package alchemist.util

import scala.collection.JavaConverters._

import joptsimple.{OptionSet, OptionParser}

class ArgParser() {

  val parser = new OptionParser()
  parser.allowsUnrecognizedOptions()
  var optionSet: OptionSet = _

  def addOptionToParser(option: Tuple4[String, String, String, Boolean]): Unit = {
    val opt: String = option._1
    val desc: String = option._2
    val clas: String = option._3
    val required: Boolean = option._4
    // add option to parser
    clas match {
      case "String" => {
        if (required) parser.accepts(opt, desc).withRequiredArg().ofType(classOf[String])
        else  parser.accepts(opt, desc).withOptionalArg().ofType(classOf[String])
      }
      case "Int" => {
        if (required) parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Int])
        else  parser.accepts(opt, desc).withOptionalArg().ofType(classOf[Int])
      }
      case "Long" => {
        if (required) parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Long])
        else  parser.accepts(opt, desc).withOptionalArg().ofType(classOf[Long])
      }
      case "Double" => {
        if (required) parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Double])
        else  parser.accepts(opt, desc).withOptionalArg().ofType(classOf[Double])
      }
      case "Boolean" => {
        if (required) parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Boolean])
        else  parser.accepts(opt, desc).withOptionalArg().ofType(classOf[Boolean])
      }
    }
  }
 
  def addOptionsToParser(options: Tuple4[String, String, String, Boolean]*): Unit = {
    options.foreach(addOptionToParser)
  }
  
  def parse(args: Array[String]) = {
    optionSet = parser.parse(args: _*)
  }
  
  def getOptionValue[T: Manifest](option: (String, String, String, Boolean)): T = {

//    if (option._4) {
      option._3 match {
        case "String"  => optionSet.valueOf(option._1).asInstanceOf[T]
        case "Int"     => optionSet.valueOf(option._1).asInstanceOf[T]
        case "Long"    => optionSet.valueOf(option._1).asInstanceOf[T]
        case "Double"  => optionSet.valueOf(option._1).asInstanceOf[T]
        case "Boolean" => optionSet.valueOf(option._1).asInstanceOf[T]
      }
//    }
//    else {
//      option._3 match {
//        case "String"  => " ".asInstanceOf[T]
//        case "Int"     => 0.asInstanceOf[T]
//        case "Long"    => 0L.asInstanceOf[T]
//        case "Double"  => 0.0.asInstanceOf[T]
//        case "Boolean" => false.asInstanceOf[T]
//    }
  }

  def intOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[Int]

  def stringOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[String]

  def booleanOptionValue(option: (String, String)) =
    optionSet.has(option._1)

  def doubleOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[Double]

  def longOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[Long]

  def optionValue[T](option: String) =
    optionSet.valueOf(option).asInstanceOf[T]

  def getOptions: Map[String, String] = {
    optionSet.asMap().asScala.flatMap { case (spec, values) =>
      if (spec.options().size() == 1 && values.size() == 1) {
        Some((spec.options().iterator().next(), values.iterator().next().toString))
      } else {
        None
      }
    }.toMap
  }
}