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

	unsigned size = 64;
	M4D::Imaging::Image< uint8, 3 >::Ptr image =
	M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped< uint8 >( size, size, size );
	for( unsigned i=0; i<size; ++i ) {
		for( unsigned j=0; j<size; ++j ) {
			for ( unsigned k=0; k<size; ++k ) {
				//image->GetElement( i, j ) = ((i>>4)+(j>>4)) & 1 ? 0 : 255;
				//image->GetElement( i, j ) = (/*(i>>4)+*/(j)) & 0xf ? 0 : 255;
				image->GetElement( Vector< int32, 3 >( i, j, k ) ) = (PWR(i-size/2)+PWR(j-size/2)+PWR(k-size/2)) < PWR(size/4) ? 0 : 255;
			}
		}
	}

	std::cout << "Saving...";
	M4D::Imaging::ImageFactory::DumpImage( fileName, *image );
	std::cout << "Done\n";
	
	std::cout << "Finished.\n";
	return 0;
}

