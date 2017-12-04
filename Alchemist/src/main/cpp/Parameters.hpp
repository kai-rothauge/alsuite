#ifndef ALCHEMIST__PARAMETERS_HPP
#define ALCHEMIST__PARAMETERS_HPP

#include <string>
#include <map>
#include <memory>

using std::string;

struct Parameter {
public:
	Parameter(string name, string type) :
		_name(name), _type(type) {}

	~Parameter() {}

	string get_name() const {
		return _name;
	}

	string get_type() const {
		return _type;
	}

	virtual string to_string() const = 0;
protected:
	string _name;
	string _type;
};

struct IntParameter : Parameter {
public:

	IntParameter(string name, int value) :
		Parameter(name, "i"), _value(value) {}

	~IntParameter() {}

	int get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	int _value;
};

struct LongParameter : Parameter {
public:

	LongParameter(string name, long value) :
		Parameter(name, "l"), _value(value) {}

	~LongParameter() {}

	long get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	long _value;
};

struct LongLongParameter : Parameter {
public:

	LongLongParameter(string name, long long value) :
		Parameter(name, "ll"), _value(value) {}

	~LongLongParameter() {}

	long long get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	long long _value;
};

struct UnsignedParameter : Parameter {
public:

	UnsignedParameter(string name, unsigned value) :
		Parameter(name, "u"), _value(value) {}

	~UnsignedParameter() {}

	unsigned get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	unsigned _value;
};

struct UnsignedLongParameter : Parameter {
public:

	UnsignedLongParameter(string name, unsigned long value) :
		Parameter(name, "ul"), _value(value) {}

	~UnsignedLongParameter() {}

	unsigned long get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	unsigned long _value;
};

struct UnsignedLongLongParameter : Parameter {
public:

	UnsignedLongLongParameter(string name, unsigned long long value) :
		Parameter(name, "ull"), _value(value) {}

	~UnsignedLongLongParameter() {}

	unsigned long long get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	unsigned long long _value;
};

struct FloatParameter : Parameter {
public:

	FloatParameter(string name, float value) :
		Parameter(name, "f"), _value(value) {}

	~FloatParameter() {}

	float get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	float _value;
};

struct DoubleParameter : Parameter {
public:

	DoubleParameter(string name, double value) :
		Parameter(name, "d"), _value(value) {}

	~DoubleParameter() {}

	double get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	double _value;
};

struct LongDoubleParameter : Parameter {
public:

	LongDoubleParameter(string name, long double value) :
		Parameter(name, "ld"), _value(value) {}

	~LongDoubleParameter() {}

	long double get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	long double _value;
};

struct CharParameter : Parameter {
public:

	CharParameter(string name, char value) :
		Parameter(name, "c"), _value(value) {}

	~CharParameter() {}

	char get_value() const {
		return _value;
	}

	string to_string() const {
		return string(1, _value);
	}

protected:
	char _value;
};

struct BoolParameter : Parameter {
public:

	BoolParameter(string name, bool value) :
		Parameter(name, "b"), _value(value) {}

	~BoolParameter() {}

	bool get_value() const {
		return _value;
	}

	string to_string() const {
		return _value ? "t" : "f";
	}

protected:
	bool _value;
};

struct StringParameter : Parameter {
public:

	StringParameter(string name, string value) :
		Parameter(name, "s"), _value(value) {}

	~StringParameter() {}

	string get_value() const {
		return _value;
	}

	string to_string() const {
		return _value;
	}

protected:
	string _value;
};

struct MatrixHandleParameter : Parameter {
public:

	MatrixHandleParameter(string name, int value) :
		Parameter(name, "mh"), _value(value) {}

	~MatrixHandleParameter() {}

	int get_value() const {
		return _value;
	}

	string to_string() const {
		return std::to_string(_value);
	}

protected:
	int _value;
};

struct Parameters {
public:
	Parameters() {}

	~Parameters() {}

	void add(Parameter * p) {
		_parameters.insert(std::make_pair(p->get_name(), p));
	}

	int num() const {
		return _parameters.size();
	}

	std::shared_ptr<Parameter> get(string name) const {
		return _parameters.find(name)->second;
	}

	int get_int(string name) const {
		return std::dynamic_pointer_cast<IntParameter> (_parameters.find(name)->second)->get_value();
	}

	float get_long(string name) const {
		return std::dynamic_pointer_cast<LongParameter> (_parameters.find(name)->second)->get_value();
	}

	float get_longlong(string name) const {
		return std::dynamic_pointer_cast<LongLongParameter> (_parameters.find(name)->second)->get_value();
	}

	float get_unsigned(string name) const {
		return std::dynamic_pointer_cast<UnsignedParameter> (_parameters.find(name)->second)->get_value();
	}

	float get_unsignedlong(string name) const {
		return std::dynamic_pointer_cast<UnsignedLongParameter> (_parameters.find(name)->second)->get_value();
	}

	float get_unsignedlonglong(string name) const {
		return std::dynamic_pointer_cast<UnsignedLongLongParameter> (_parameters.find(name)->second)->get_value();
	}

	float get_float(string name) const {
		return std::dynamic_pointer_cast<FloatParameter> (_parameters.find(name)->second)->get_value();
	}

	double get_double(string name) const {
		return std::dynamic_pointer_cast<DoubleParameter> (_parameters.find(name)->second)->get_value();
	}

	double get_longdouble(string name) const {
		return std::dynamic_pointer_cast<LongDoubleParameter> (_parameters.find(name)->second)->get_value();
	}

	string get_string(string name) const {
		return std::dynamic_pointer_cast<StringParameter> (_parameters.find(name)->second)->get_value();
	}

	char get_char(string name) const {
		return std::dynamic_pointer_cast<CharParameter> (_parameters.find(name)->second)->get_value();
	}

	bool get_bool(string name) const {
		return std::dynamic_pointer_cast<BoolParameter> (_parameters.find(name)->second)->get_value();
	}

	bool get_matrix_handle(string name) const {
		return std::dynamic_pointer_cast<MatrixHandleParameter> (_parameters.find(name)->second)->get_value();
	}
//
//	void * get_ptr(string name) const {
//		return static_cast< Parameter<void *> *>(_parameters.find(name)->second)->get_value();
//	}

	string to_string() const {
		string arg_list = "";

		for (auto it = _parameters.begin(); it != _parameters.end(); it++ ) {
			arg_list.append(it->first);
			arg_list.append("(");
			arg_list.append(it->second->get_type());
			arg_list.append(")");
			arg_list.append(it->second->to_string());
			arg_list.append(" ");
		}
		return arg_list;
	}

private:
	std::map<string, std::shared_ptr<Parameter> > _parameters;
};

#endif
