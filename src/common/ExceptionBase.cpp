#include "ExceptionBase.h"

#include "Log.h"
#include "Debug.h"

namespace M4D
{
namespace ErrorHandling
{


ExceptionBase::ExceptionBase( char * name )
		: std::exception(), _name( name ) 
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
