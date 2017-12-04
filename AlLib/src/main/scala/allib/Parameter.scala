package allib

class Parameter(val _name: String, val _ctype: String, val _value: String) {
  
  
  
  def getName(): String = _name
  
  def getType(): String = _ctype
  
  def getValue(): String = _value
  
  override def toString(): String = _name + "(" + _ctype + "):" + _value
}

object Parameter {
  
  def apply(_name: String, _ctype: String, _value: String = ""): Parameter = new Parameter(_name, _ctype, _value)
}

