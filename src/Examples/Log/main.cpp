
#include "Log.h"

#include <iomanip>

int
main( void )
{
	LOG << LogDelimiter( '=', 80 );
	LOG << "This is example text" << std::endl;
	LOG << LogDelimiter( '=', 80 );

	return 0;
}
