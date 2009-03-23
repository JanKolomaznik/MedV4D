/**
 *  @ingroup common
 *  @file ExceptionBase.cpp
 *  @author Jan Kolomaznik
 */
#include "ExceptionBase.h"

#include "Log.h"
#include "Debug.h"

namespace M4D
{
namespace ErrorHandling
{


ExceptionBase::ExceptionBase( std::string name ) throw()
		: std::exception(), _name( name ) 
#ifdef DEBUG_LEVEL
		,_exceptionFileName( ____EXCEPTION_FILE_NAME ), 
		_exceptionLineNumber( ____EXCEPTION_LINE_NUMBER )
#endif /*DEBUG_LEVEL*/
{ 

#ifdef DEBUG_LEVEL
	ResetExceptionInfo();
#endif /*DEBUG_LEVEL*/

	OnRaise();
}

ExceptionBase::ExceptionBase() throw()
	: std::exception(), _name( "General exception raised." ) 
#ifdef DEBUG_LEVEL
		,_exceptionFileName( ____EXCEPTION_FILE_NAME ), 
		_exceptionLineNumber( ____EXCEPTION_LINE_NUMBER )
#endif /*DEBUG_LEVEL*/
{ 

#ifdef DEBUG_LEVEL
	ResetExceptionInfo();
#endif /*DEBUG_LEVEL*/

	OnRaise();
}

void
ExceptionBase::OnRaise()
{ 
	DS_PRINT( _exceptionFileName << ":" << _exceptionLineNumber << " Raised exception : " << (*this) );
}


std::ostream& operator<<( std::ostream &out, ExceptionBase &exception )
{
	out << exception.what();
	return out;
}

} /*namespace ErrorHandling*/
} /*namespace M4D*/
