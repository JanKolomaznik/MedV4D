#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>
#include <QApplication>
#include "ViewerWindow.h"

int main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	//////////////////////////////////////////////////////////////////////////
	// Viewer Testing
	//////////////////////////////////////////////////////////////////////////

	if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];


	std::cout << "Loading file...";
	M4D::Imaging::AbstractImage::Ptr image = 
		M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AbstractImage > prodconn;
	prodconn.PutDataset( image );


	QApplication app(argc, argv);
	ViewerWindow viewer(prodconn);
	viewer.show();
	return app.exec();

	////////////////////////////////////////////////////////////////////////////
	//// GENERATING TESTING DATA
	////////////////////////////////////////////////////////////////////////////
	//
	//if( argc < 2 || argc > 2 ) {
	//	std::cerr << "Wrong argument count - must be in form: 'program outputfile'\n";
	//	return 1;
	//}

	//std::string fileName = argv[1];

	//unsigned size = 32;
	//unsigned hSize = size / 2;
	//unsigned qSize = hSize / 2;

	//M4D::Imaging::Image< int16, 3 >::Ptr image =
	//M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped< int16 >(size, size, size);

	//float var = 4.0;

	//for( unsigned i=0; i<size; ++i ) {
	//	for( unsigned j=0; j<size; ++j ) {
	//		for (unsigned k=0; k<size; ++k) {
	//			//image->GetElement( i, j ) = ((i>>4)+(j>>4)) & 1 ? 0 : 255;
	//			//image->GetElement( i, j ) = (/*(i>>4)+*/(j)) & 0xf ? 0 : 255;
	//			image->GetElement( Vector< int32, 3 >( i, j, k) ) = (PWR(i-hSize)+PWR(j-hSize)+PWR(k-hSize)) < PWR(qSize) ? (int)((sqrt((float)(PWR(i-hSize)+PWR(j-hSize)+PWR(k-hSize)))) * var) : 0;
	//		}
	//	}
	//}

	//std::cout << "Saving...";
	//M4D::Imaging::ImageFactory::DumpImage( fileName, *image );
	//std::cout << "Done\n";

	//std::cout << "Finished.\n";
	//return 0; 
} 