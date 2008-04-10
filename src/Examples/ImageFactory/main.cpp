#include "Log.h"
#include "ImageFactory.h"

using namespace M4D::Images;

int
main( void )
{
	AbstractImage::APtr p;
	ImageDataTemplate<int>::Ptr p2;

	LOG ( LogDelimiter( '=', 80 ) );
	LOG ( "Creating 2D image with int elements... " << std::endl );

	p = ImageFactory::CreateEmptyImage2D<int>( 100, 100 );

	p2 = boost::dynamic_pointer_cast<ImageDataTemplate<int>, AbstractImage >( p );

	p.reset();

	/*LOG ( (p.use_count()) );
	LOG ( (p2.use_count()) );*/

	p2->Get( 100 ) = 235;

	LOG ( "...image created. " );

	LOG ( LogDelimiter( '=', 80 ) );

	return 0;
}
