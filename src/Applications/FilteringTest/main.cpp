
#include "Common.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/DefaultConnection.h"
#include <iostream>


using namespace M4D::Imaging;
using namespace std;


typedef OutImageConnectionSimple< Image< int16, 3 > > OutConn;
typedef InImageConnectionSimple< Image< int16, 3 > > InConn;

int
main( int argc, char** argv )
{
	LOG( "** STARTING FILTERING TESTS **" );
	TEST_FILTER	filter;
	OutConn	oconn;
	InConn iconn;

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
