#include "Imaging/ImageFactory.h"
#include "common/Common.h"
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

	/*unsigned size = 256;
	M4D::Imaging::Image< uint32, 3 >::Ptr image =
	M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped< uint32 >( size, size, size );
	for( unsigned i=0; i<size; ++i ) {
		for( unsigned j=0; j<size; ++j ) {
			for ( unsigned k=0; k<size; ++k ) {
				image->GetElement( Vector< int32, 3 >( i, j, k ) ) = Abs(5000 + -1*ClampToInterval<uint32>( 10, 5000, Abs( Sqr(i-128) + Sqr(j-128) -2000 ) ) );
			}
		}
	}*/

	unsigned size = 256;
	unsigned hsize = size / 2;
	float step = 5.0f / size;
	M4D::Imaging::Image< Vector2f, 2 >::Ptr image =
		M4D::Imaging::ImageFactory::CreateEmptyImage2DTyped< Vector2f >( size, size );
	for( unsigned i=0; i<size; ++i ) {
		for( unsigned j=0; j<size; ++j ) {
			float x = (i - hsize) * step;
			float y = (j - hsize) * step;
			image->GetElement( Vector< int32, 2 >( i, j ) ) = Vector2f( 
							Sqr( sin( x ) ) + y, 
							cos( x + y*y ) 
						);
		}
	}

	std::cout << "Saving...";
	M4D::Imaging::ImageFactory::RawDumpImage( fileName, *image, std::cout );
	std::cout << "Done\n";
	
	std::cout << "Finished.\n";
	return 0;
}

