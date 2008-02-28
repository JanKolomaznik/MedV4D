//Presentation of debugging macros.
//If you want to run program in debug mode you define DEBUG_LEVEL="requested level"
//in compiler parameters. If you wish to have additional informations in debugging prints
//define DEBUG_ADITIONAL_INFO in compiler parameters.
//Definition of these macros in source code isn't recommended.

#include "Debug.h"

#include <iostream>
#include <iomanip>

//Example class, for which we create special printing function.
class DebugClass
{
public:
	DebugClass( int val ): value( val ) {}
	int value;
};

//Extending of printing functions set. We can easily add custom printing function for our objects.
std::ostream & operator<<( std::ostream &outStream, const DebugClass & printedObject )
{
	outStream << "======================================\nTest object contains value : " << printedObject.value 
		<< "\n======================================" << std::endl;
	return outStream;
}

int
main( void )
{
	std::cout << "Starting example program for debug prints." << std::endl << std::endl;

	//Basic debug print.
	D_PRINT( "Testing debug print N.1" << std::endl  << std::endl );	

	//Debug prints in different levels.
	for( int level=0; level<10; ++level ) {
		DL_PRINT( level, "Printing at debug level : " << level << std::endl );
	}
	
	std::cout << std::endl;
	
	//Use of debug command - executed only in debug mode.
	D_COMMAND( std::cout << "This text is printed only in debug mode" << std::endl );

	for( int level=0; level<10; ++level ) {
		DL_COMMAND( level, std::cout << "This text is printed only in debug mode, if debug level is greater than or equal to " << level << "." << std::endl );
	}

	std::cout << std::endl;

	//Debug print of custom class.
	D_PRINT( DebugClass( 666 ) );
	
	std::cout << std::endl;

	std::cout << "Let's try assertion..." << std::endl; 

	//Try assertion.
	ASSERT( 100<50 );
	return 0;
}
