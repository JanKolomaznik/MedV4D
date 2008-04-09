#include "Log.h"
#include "ImageFactory.h"

using namespace M4D::Images;

int
main( void )
{
	AbstractImage::Ptr p;

	LOG ( LogDelimiter( '=', 80 ) );
	LOG ( "Creating 2D image with int elements... " << std::endl );
	p = ImageFactory::CreateEmptyImage2D<int>( 100, 100 );

	LOG ( "...image created. " << p << std::endl );

	LOG ( LogDelimiter( '=', 80 ) );

	return 0;
}
