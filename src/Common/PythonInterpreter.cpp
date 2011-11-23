#include "MedV4D/Common/PythonInterpreter.h"

#ifdef USE_PYTHON

#include <boost/python.hpp>
using namespace boost::python;

std::string
currentException()
{
	PyObject *exc,*val,*tb;
	PyErr_Fetch(&exc,&val,&tb);
	handle<> hexc(exc);
	if(!tb || !val)
	{
		ASSERT(hexc);
		return extract<std::string>(str(hexc));
	}
	handle<> hval(val), htb(tb);

	object traceback(import("traceback"));
	object format_exception(traceback.attr("format_exception"));
	object formatted_list(format_exception(hexc,hval,htb));    
	object formatted(str("\n").join(formatted_list));
	return extract<std::string>(formatted);
}



struct PythonOutputRedirection
{
	PythonOutputRedirection()
	{
		LOG( "constructing redirection" )
	}
	void
	write( std::string text )
	{
		ASSERT( mPrinter )
		(*mPrinter)( text );
	}
	
	Printer::Ptr mPrinter;
};




PythonInterpreter::PythonInterpreter()
{
	Py_Initialize();

	try {
		mMainModule = boost::python::import("__main__");
		mMainNamespace = mMainModule.attr("__dict__");

		//initPrintRedirection();
		//boost::python::exec( "import PrintRedirection\n", mMainNamespace ); 
		//mMainNamespace[ "redirect" ] = class_<PythonOutputRedirection>("PythonOutputRedirection")
        	//	.def("write", &PythonOutputRedirection::write)();
		
		

	} catch(boost::python::error_already_set const &)
	{
		//PyErr_Print();
		errPrint( currentException() );
	}
}

void
PythonInterpreter::exec( const std::string &cmd )
{
	try {
		boost::python::object result = boost::python::exec(
				cmd.data(),
				mMainNamespace
				);
		boost::python::extract<std::string> resString( result );
		if ( resString.check() ) {
			print( resString() );
		}
	} catch(boost::python::error_already_set const &)
	{
		errPrint( currentException() );
	}

}


void
PythonInterpreter::setStdPrinter( Printer::Ptr aPrinter )
{
	ASSERT( aPrinter );

	mStdPrinter = aPrinter;

	object tmp1 = class_<PythonOutputRedirection>("PythonOutputRedirection")
        		.def("write", &PythonOutputRedirection::write)();


	extract<PythonOutputRedirection&> extr( tmp1 );
	ASSERT( extr.check() )
	PythonOutputRedirection &redir = extr();
	redir.mPrinter = aPrinter;
	
	object sysModule = boost::python::import("sys");
	object sysNamespace = sysModule.attr("__dict__");

	sysNamespace["stdout"] = tmp1;
}

	void
PythonInterpreter::setErrPrinter( Printer::Ptr aPrinter )
{
	ASSERT( aPrinter );

	mErrPrinter = aPrinter;

	object tmp1 = class_<PythonOutputRedirection>("PythonOutputRedirection")
        		.def("write", &PythonOutputRedirection::write)();


	extract<PythonOutputRedirection&> extr( tmp1 );
	ASSERT( extr.check() )
	PythonOutputRedirection &redir = extr();
	redir.mPrinter = aPrinter;
	
	object sysModule = boost::python::import("sys");
	object sysNamespace = sysModule.attr("__dict__");

	sysNamespace["stderr"] = tmp1;
}


#endif /*USE_PYTHON*/
