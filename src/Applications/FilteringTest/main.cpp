
#include "Common.h"
#include "Imaging/AbstractFilter.h"
#include <iostream>


using namespace M4D::Imaging;
using namespace std;

int
main( int argc, char** argv )
{
	LOG( "** STARTING FILTERING TESTS **" );
	TEST_FILTER	filter;

	filter.Execute();

	char option = '\0';
	cin >> option;
	while ( option != 'q' && option != 'Q' ) {
		switch( option ) {

		case 's':
		case 'S':
			LOG( "STOP CALLED" );
			filter.StopExecution();
			break;

		default:
			break;
		}

		cin >> option;
	}

	LOG( "** FILTERING TESTS FINISHED **" )
	return 0;
}
