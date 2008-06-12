
#include "Common.h"
#include "Imaging/BasicImageFilters.h"
#include "Imaging/DefaultConnection.h"
#include "Imaging/ImageFactory.h"
#include <iostream>


using namespace M4D::Imaging;
using namespace std;


typedef Image< int16, 3 > ImageType;
typedef ImageConnectionSimple< ImageType > Conn;
typedef ImageConnectionSimple< ImageType > Conn;

int
main( int argc, char** argv )
{
	LOG( "** STARTING FILTERING TESTS **" );
	CopyImageFilter< ImageType, ImageType > filter;
	Conn	oconn;
	Conn iconn;

	ImageType::Ptr inputImage = ImageFactory::CreateEmptyImage3DTyped< int16 >( 50,50,50 );
	ImageType::Ptr outputImage = ImageFactory::CreateEmptyImage3DTyped< int16 >( 50,50,50 );

	oconn.ConnectOut( filter.InputPort()[0] );
	iconn.ConnectIn( filter.OutputPort()[0] );

	oconn.PutImage( inputImage );
	iconn.PutImage( outputImage );
	

	char option = '\0';
	cin >> option;
	while ( option != 'q' && option != 'Q' ) {
		switch( option ) {
		case 'e':
		case 'E':
			filter.Execute();
			break;
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
