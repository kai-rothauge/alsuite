#include "Command.hpp"

namespace alchemist {


//
//void RunTaskCommand::run(Worker * e) const {
//
//	typedef int (*run_t)(std::string, const Parameters &, Parameters &);
//
//	run_t run_f = (run_t) sym;
//	run_f(task_name, input_parameters, output_parameters);
//
//	e->log->info("Finished call to {}::run", library_name);
//}
//
//void HaltCommand::run(Worker * e) const {
//	e->shouldExit = true;
//}

}
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::Command);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::LoadLibraryCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::RunTaskCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::HaltCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::NewMatrixCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixMulCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::MatrixGetRowsCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::ThinSVDCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::TransposeCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::KMeansCommand);
BOOST_CLASS_EXPORT_IMPLEMENT(alchemist::TruncatedSVDCommand);




