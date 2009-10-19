
#include "Log.h"

#include <iomanip>

int
main( void )
{
	LOG ( LogDelimiter( '=', 80 ) );
	LOG ( "This is example text. Time : " << LogCurrentTime() << std::endl );
	LOG ( LogDelimiter( '=', 80 ) );

	return 0;
}
