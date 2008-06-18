
#include "Common.h"
#include "Imaging/ExampleImageFilters.h"
#include "Imaging/DefaultConnection.h"
#include "Imaging/ImageFactory.h"
#include <iostream>


using namespace M4D::Imaging;
using namespace std;


typedef Image< int16, 3 > Image3DType;
typedef Image< int16, 2 > Image2DType;
typedef ImageConnectionSimple< Image3DType > ProducerConn;
typedef ImageConnectionSimple< Image2DType > ConsumerConn;
typedef ImageConnectionImOwner< Image3DType > InterConn;

int
main( int argc, char** argv )
{
	LOG( "** STARTING FILTERING TESTS **" );
	CopyImageFilter< Image3DType, Image3DType > copyfilter;
	ColumnMaxImageFilter< int16 > maxfilter;
	ProducerConn prodconn;
	ConsumerConn consconn;
	InterConn interconn;

	Image3DType::Ptr inputImage = ImageFactory::CreateEmptyImage3DTyped< int16 >( 50,50,50 );
	Image2DType::Ptr outputImage = ImageFactory::CreateEmptyImage2DTyped< int16 >( 10,10 );

	prodconn.ConnectConsumer( copyfilter.InputPort()[0] );
	interconn.ConnectProducer( copyfilter.OutputPort()[0] );
	interconn.ConnectConsumer( maxfilter.InputPort()[0] );
	consconn.ConnectProducer( maxfilter.OutputPort()[0] );

	prodconn.PutImage( inputImage );
	consconn.PutImage( outputImage );
	

	char option = '\0';
	cin >> option;
	while ( option != 'q' && option != 'Q' ) {
		switch( option ) {
		case 'e':
		case 'E':
			copyfilter.Execute();
			break;
		case 's':
		case 'S':
			LOG( "STOP CALLED" );
			copyfilter.StopExecution();
			break;

		default:
			break;
		}

		cin >> option;
	}

	LOG( "** FILTERING TESTS FINISHED **" )

	
	return 0;
}
