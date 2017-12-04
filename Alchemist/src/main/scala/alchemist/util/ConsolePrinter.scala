package alchemist.util

import java.io.PrintWriter

object ConsolePrinter {
  
  var consoleColor: String = _
  var consoleTab: String = ""
  var logWriter: PrintWriter = _
  var errWriter: PrintWriter = _
  
  var tabs: Int = 0
  
  def tab {
    tabs = tabs + 1
    consoleTab = consoleTab + "    "
  }
  
  def untab {
    tabs = 0.max(tabs-1)
    consoleTab = ""
    for (i <- 1 to tabs) {
      consoleTab = consoleTab + "    "
    }
  }
  
  def setLogWriter(arg: PrintWriter) {
    logWriter = arg
  }
  
  def setErrWriter(arg: PrintWriter) {
    errWriter = arg
  }
    
  def setColor(arg: String) {
    consoleColor = formatConsoleColor(arg)
  }
    
  def println(text: Any, args: Any*) {
    if (logWriter != null) logWriter.write(consoleTab + text.toString + "\n")
    printf(consoleTab + consoleColor + text.toString + Console.RESET + "\n")
  }
    
  def printError(text: Any, args: Any*) {
    if (errWriter != null) errWriter.write(consoleTab + text.toString + "\n")
    printf(consoleTab + consoleColor + text.toString + Console.RESET + "\n")
  }

  def formatConsoleColor(arg: String): String = {
    arg match {
      case "cyan"           => Console.CYAN
      case "white"          => Console.WHITE
      case "black"          => Console.BLACK  
      case "blue"           => Console.BLUE
      case "red"            => Console.RED
      case "green"          => Console.GREEN
      case "yellow"         => Console.YELLOW
      case "magenta"        => Console.MAGENTA
      case "bright cyan"    => Console.BOLD + Console.CYAN
      case "bright white"   => Console.BOLD + Console.WHITE
      case "bright black"   => Console.BOLD + Console.BLACK  
      case "bright blue"    => Console.BOLD + Console.BLUE
      case "bright red"     => Console.BOLD + Console.RED
      case "bright green"   => Console.BOLD + Console.GREEN
      case "bright yellow"  => Console.BOLD + Console.YELLOW
      case "bright magenta" => Console.BOLD + Console.MAGENTA
      case _                => Console.BOLD + Console.WHITE
    } 
  }
}