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
{ 
	OnRaise();
}

ExceptionBase::ExceptionBase() throw()
	: std::exception(), _name( "General exception raised." ) 
{ 
	OnRaise();
}

void
ExceptionBase::OnRaise()
{ 
	DL_PRINT( EXCEPTION_DEBUG_LEVEL, "Raised exception : " << (*this) );
}


std::ostream& operator<<( std::ostream &out, ExceptionBase &exception )
{
	out << exception.what();
	return out;
}

} /*namespace ErrorHandling*/
} /*namespace M4D*/
