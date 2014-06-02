#ifndef PYTHON_INTERPRETER_H
#define PYTHON_INTERPRETER_H

#ifdef USE_PYTHON


#include "MedV4D/Common/Common.h"

#include <boost/python.hpp>

struct Printer
{
	typedef std::shared_ptr< Printer > Ptr;

	virtual void
	operator()( const std::string &aText ) = 0;
};


class PythonInterpreter
{
public:
	PythonInterpreter();
	

	void
	exec( const std::string &cmd );

	void
	setStdPrinter( Printer::Ptr aPrinter );

	void
	setErrPrinter( Printer::Ptr aPrinter );

protected:
	void 
	print( std::string aText )
	{
		if( mStdPrinter ) {
				(*mStdPrinter)( aText );
		} else {
			LOG( aText );
		}
	}
	void 
	errPrint( std::string aText )
	{
		if( mErrPrinter ) {
				(*mErrPrinter)( aText );
		} else {
			LOG( aText ); //TODO
		}
	}


	boost::python::object mMainModule;
	boost::python::object mMainNamespace;

	Printer::Ptr mStdPrinter;
	Printer::Ptr mErrPrinter;
};


#endif /*USE_PYTHON*/
#endif /*PYTHON_INTERPRETER_H*/
