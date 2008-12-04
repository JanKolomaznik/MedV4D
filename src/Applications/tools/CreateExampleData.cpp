#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

int
main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program outputfile'\n";
		return 1;
	}

	std::string fileName = argv[1];

	unsigned size = 256;
	M4D::Imaging::Image< uint8, 2 >::Ptr image =
	M4D::Imaging::ImageFactory::CreateEmptyImage2DTyped< uint8 >( size, size );

	for( unsigned i=0; i<size; ++i ) {
		for( unsigned j=0; j<size; ++j ) {
			//image->GetElement( i, j ) = ((i>>4)+(j>>4)) & 1 ? 0 : 255;
			//image->GetElement( i, j ) = (/*(i>>4)+*/(j)) & 0xf ? 0 : 255;
			image->GetElement( i, j ) = (PWR(i-128)+PWR(j-128)) < PWR(64) ? 0 : 255;
		}
	}

	std::cout << "Saving...";
	M4D::Imaging::ImageFactory::DumpImage( fileName, *image );
	std::cout << "Done\n";
	
	std::cout << "Finished.\n";
	return 0;
}

