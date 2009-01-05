#include "Common.h"
#include "Filtering.h"
#include "Imaging/Image.h"


using namespace M4D;
using namespace M4D::Imaging;

typedef Image< int16, 3 > ImageType;

template< typename ElementType >
M4D::Imaging::AbstractImage::AImagePtr
CropImage( typename M4D::Imaging::Image<ElementType, 3>::Ptr image )
{
	return image->GetRestrictedImage( image->GetRegion().GetSlice( 5 ) );
}

int
main( int argc, char **argv )
{

	if( argc < 3 || argc > 3 ) {
                std::cerr << "Wrong argument count - must be in form: 'program inputfile outputfile'\n";
                return 1;
        }

	std::string inFilename = argv[1];
	std::string outFilename = argv[2];


	std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AbstractImage::AImagePtr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";

	M4D::Imaging::AbstractImage::AImagePtr outImage;
	
	//IMAGE_TYPE_PTR_SWITCH_MACRO( image, outImage = CropImage( IMAGE ) );
	
	TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(), outImage = CropImage<TTYPE>( M4D::Imaging::Image<TTYPE, 3>::CastAbstractImage( image ) ) );

	std::cout << "Saving file..."; std::cout.flush();
	M4D::Imaging::ImageFactory::DumpImage( outFilename, *outImage );
	std::cout << "Done\n";

	return 0;
}

