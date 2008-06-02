
#include "Common.h"
#include "Imaging/BasicImageFilters.h"
#include "Imaging/DefaultConnection.h"
#include <iostream>


using namespace M4D::Imaging;
using namespace std;


typedef Image< int16, 3 > ImageType;
typedef OutImageConnectionSimple< ImageType > OutConn;
typedef InImageConnectionSimple< ImageType > InConn;

int
main( int argc, char** argv )
{
	LOG( "** STARTING FILTERING TESTS **" );
	CopyImageFilter< ImageType, ImageType > filter;
	OutConn	oconn;
	InConn iconn;

	oconn.ConnectOut( filter.InputPort()[0] );
	iconn.ConnectIn( filter.OutputPort()[0] );

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
