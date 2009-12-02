#include "common/Common.h"
#include "Filtering.h"
#include "Imaging/Image.h"
#undef min
#undef max
#include <tclap/CmdLine.h>


using namespace M4D;
using namespace M4D::Imaging;

typedef Image< int16, 3 > ImageType;

template< typename ElementType >
M4D::Imaging::AImage::Ptr
GetSlice( typename M4D::Imaging::Image<ElementType, 3>::Ptr image, int slice )
{
	return image->GetRestrictedImage( image->GetRegion().GetSlice( slice ) );
}

template< typename ImageType >
M4D::Imaging::AImage::Ptr
CropImage( typename ImageType::Ptr image, const std::vector<int> &firstCorner, const std::vector<int> &secondCorner )
{
	typedef Vector< int, ImageTraits< ImageType >::Dimension > CornerType;

	if( firstCorner.size() != secondCorner.size() && firstCorner.size() != ImageTraits< ImageType >::Dimension ) {
		throw M4D::ErrorHandling::EBadDimension();
	}
	CornerType p1(0);
	CornerType p2(0);

	for( unsigned i = 0; i < ImageTraits< ImageType >::Dimension; ++i ) {
		p1[i] = Min( firstCorner[i], secondCorner[i] );
		p2[i] = Max( firstCorner[i], secondCorner[i] );
	}

	return image->GetRestrictedImage( image->GetSubRegion( p1, p2 ) );
}


int
main( int argc, char **argv )
{
	try{  

	TCLAP::CmdLine cmd("Tool for croping multidimensional images.", ' ', "");

	TCLAP::ValueArg<int> sliceNumber( "s", "slice", "Slice number to be kept.", false, 0, "Slice index" );
	cmd.add( sliceNumber );

	TCLAP::MultiArg< int > firstPoint( "l", "leftCorner", "First corner of croped area", false, "List of coordinates" );
	cmd.add( firstPoint );
	TCLAP::MultiArg< int > secondPoint( "r", "rightCorner", "Second corner of croped area", false, "List of coordinates" );
	cmd.add( secondPoint );

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	//TCLAP::ValueArg<std::string> inFilenameArg( "i", "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
	//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	std::string inFilename = inFilenameArg.getValue();
	std::string outFilename = outFilenameArg.getValue();


	std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";

	//M4D::Imaging::AImage::Ptr outImage;
	
	if( sliceNumber.isSet() ) {
		if( image->GetDimension() > 2 ) {
			int slice = sliceNumber.getValue();
			//IMAGE_TYPE_PTR_SWITCH_MACRO( image, outImage = CropImage( IMAGE ) );
			
			TYPE_TEMPLATE_SWITCH_MACRO( 
					image->GetElementTypeID(), 
					image = GetSlice<TTYPE>( M4D::Imaging::Image<TTYPE, 3>::CastAImage( image ), slice ) 
					);
		} else {
			throw M4D::ErrorHandling::EBadDimension();
		}
	}
	if( firstPoint.isSet() && secondPoint.isSet() ) {
		IMAGE_TYPE_PTR_SWITCH_MACRO( image, 
				image = CropImage< IMAGE_TYPE >( IMAGE, firstPoint.getValue(), secondPoint.getValue() );
				);
	}

	std::cout << "Saving file..."; std::cout.flush();
	M4D::Imaging::ImageFactory::DumpImage( outFilename, *image );
	std::cout << "Done\n";

	} catch ( ... ) {
		std::cerr << "error \n";
	}

	return 0;
}

