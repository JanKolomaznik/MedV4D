
#include "Common.h"
#include "Imaging/ExampleImageFilters.h"
#include "Imaging/DefaultConnection.h"
#include "Imaging/ImageFactory.h"
#include <iostream>


using namespace M4D::Imaging;
using namespace std;


typedef Image< int16, 3 > Image3DType;
typedef Image< int16, 2 > Image2DType;
typedef ImageConnectionSimple< Image3DType > Conn3D;
typedef ImageConnectionSimple< Image2DType > Conn2D;
typedef ImageConnectionImOwner< Image3DType > InterConn3D;

int
main( int argc, char** argv )
{
	LOG( "** STARTING FILTERING TESTS **" );
	CopyImageFilter< Image3DType, Image3DType > copyfilter;
	//ColumnMaxImageFilter< int16 > maxfilter;
	//maxfilter.SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );
	SimpleConvolutionImageFilter< Image3DType > convolution;
	convolution.SetUpdateInvocationStyle( AbstractPipeFilter::UIS_ON_UPDATE_FINISHED );

	Conn3D prodconn;
	//Conn2D consconn;
	Conn3D consconn;
	InterConn3D interconn;

	Image3DType::Ptr inputImage = ImageFactory::CreateEmptyImage3DTyped< int16 >( 50,50,50 );
	//Image2DType::Ptr outputImage = ImageFactory::CreateEmptyImage2DTyped< int16 >( 10,10 );
	Image3DType::Ptr outputImage = ImageFactory::CreateEmptyImage3DTyped< int16 >( 10,10, 10 );

	prodconn.ConnectConsumer( copyfilter.InputPort()[0] );
	interconn.ConnectProducer( copyfilter.OutputPort()[0] );
	interconn.ConnectConsumer( convolution.InputPort()[0] );
	consconn.ConnectProducer( convolution.OutputPort()[0] );

	prodconn.PutImage( inputImage );
	consconn.PutImage( outputImage );

	cout << "E : Execute pipeline\nS : Stop execution\nQ : Quit\n";

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
