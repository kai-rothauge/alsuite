#include "Executor.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace alchemist {

using namespace El;

int Executor::load_library(std::string args) {

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(" ");
	tokenizer tok(args, sep);

	tokenizer::iterator iter = tok.begin();

	std::string library_name = *iter;
	std::string library_path = *(++iter);

	char * cstr = new char [library_path.length()+1];
	std::strcpy(cstr, library_path.c_str());

	log->info("Loading library {} located at {}", library_name, library_path);

	void * l = dlopen(cstr, RTLD_NOW);
	if (l == NULL) {
		log->info("dlopen failed: {}", dlerror());
		return -1;
	}

    libraries.insert(std::make_pair(library_name, l));

    	void * f = dlsym(libraries.find(library_name)->second, "load");
    	if (f == NULL) {
    		log->info("dlsym with command \"load\" failed: {}", dlerror());
    		return -1;
    	}

    	typedef int (*load_t)(const boost::mpi::environment &, const boost::mpi::communicator &, const boost::mpi::communicator &);

    	load_t load_f = (load_t) f;
    	load_f(env, world, peers);

    	log->info("Library {} loaded", library_name);

    	return 0;
}

int Executor::run_task(std::string args, Parameters & output_parameters) {

	log->info("Received: {}", args);

	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep(" ");
	tokenizer tok(args, sep);

	tokenizer::iterator iter = tok.begin();

	Parameters input_parameters = Parameters();
	std::string input_parameter;

	std::string library_name = *iter;
	std::string task_name = *(++iter);

	for (++iter; iter != tok.end(); ++iter) {
		input_parameter = *(iter);
		deserialize_parameters(input_parameter, input_parameters);
	}

	void * lib = libraries.find(library_name)->second;

	log->info("Calling {}::run with '{}'", library_name, task_name);

	void * f = dlsym(lib, "run");
	if (f == NULL) {
		printf("dlopen failed: %s\n", dlerror());
		return -1;
	}

	typedef int (*run_t)(std::string, const Parameters &, Parameters &, boost::mpi::environment &, boost::mpi::communicator &, boost::mpi::communicator &);

	run_t run_f = (run_t) f;
	run_f(task_name, input_parameters, output_parameters, env, world, peers);

	log->info("Finished call to {}::run with '{}'", library_name, task_name);

	return 0;
}

int Executor::unload_libraries() {

	for (auto const & lib : libraries) dlclose(lib.second);

	return 0;
}

void Executor::deserialize_parameters(std::string input_parameter, Parameters & input_parameters) {

	boost::char_separator<char> sep("()");
	boost::tokenizer<boost::char_separator<char> > tok(input_parameter, sep);
	auto tok_iter = tok.begin();

	std::string parameter_name = *tok_iter;
	std::string parameter_type = *(++tok_iter);
	std::string parameter_value = *(++tok_iter);

	if (parameter_type.compare("i") == 0)
		input_parameters.add(new IntParameter(parameter_name, std::stoi(parameter_value)));
	else if (parameter_type.compare("l") == 0)
		input_parameters.add(new LongParameter(parameter_name, std::stol(parameter_value)));
	else if (parameter_type.compare("ll") == 0)
		input_parameters.add(new LongLongParameter(parameter_name, std::stoll(parameter_value)));
	else if (parameter_type.compare("u") == 0)
		input_parameters.add(new UnsignedParameter(parameter_name, std::stoi(parameter_value)));
	else if (parameter_type.compare("ul") == 0)
		input_parameters.add(new UnsignedLongParameter(parameter_name, std::stoul(parameter_value)));
	else if (parameter_type.compare("ull") == 0)
		input_parameters.add(new UnsignedLongLongParameter(parameter_name, std::stoull(parameter_value)));
	else if (parameter_type.compare("ld") == 0)
		input_parameters.add(new LongDoubleParameter(parameter_name, std::stold(parameter_value)));
	else if (parameter_type.compare("d") == 0)
		input_parameters.add(new DoubleParameter(parameter_name, std::stod(parameter_value)));
	else if (parameter_type.compare("f") == 0)
		input_parameters.add(new FloatParameter(parameter_name, std::stof(parameter_value)));
	else if (parameter_type.compare("b") == 0)
		input_parameters.add(new BoolParameter(parameter_name, parameter_value.compare("t") == 0));
	else if (parameter_type.compare("c") == 0)
		input_parameters.add(new CharParameter(parameter_name, parameter_value[0]));
	else if (parameter_type.compare("s") == 0)
		input_parameters.add(new StringParameter(parameter_name, parameter_value));
	else if (parameter_type.compare("mh") == 0)
		input_parameters.add(new MatrixHandleParameter(parameter_name, std::stoi(parameter_value)));
}

std::string Executor::serialize_parameters(const Parameters & output) const {

	for (auto iter = libraries.begin(); iter != libraries.end(); ++iter)
		dlclose(iter->second);

	return output.to_string();
}

}
