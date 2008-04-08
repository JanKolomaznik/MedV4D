#include "ExceptionBase.h"

#include "Log.h"
#include "Debug.h"

namespace ErrorHandling
{


ExceptionBase::ExceptionBase( char * name )
		: std::exception(), _name( name ) 
{ 
	DL_PRINT( 10, "Raised exception : " << (*this) );
}



std::ostream& operator<<( std::ostream &out, ExceptionBase &exception )
{
	out << exception.what();
}

} /*namespace ErrorHandling*/
