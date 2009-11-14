#include <QApplication>
#include "ViewerWindow.h"

int main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	/*if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];


	std::cout << "Loading file...";
	M4D::Imaging::AbstractImage::Ptr image = 
		M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	*/

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AbstractImage > prodconn;
	//prodconn.PutDataset( image );


	QApplication app(argc, argv);
	ViewerWindow viewer(prodconn);
	viewer.show();
	return app.exec();
} 