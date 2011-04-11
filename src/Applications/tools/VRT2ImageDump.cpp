#include "common/Common.h"
#include "Filtering.h"
#include "Imaging/Image.h"
#undef min
#undef max
#include <tclap/CmdLine.h>
	
#include <ifstream>

int
main( int argc, char **argv )
{
	try{  

	TCLAP::CmdLine cmd("Tool for croping multidimensional images.", ' ', "");

	TCLAP::ValueArg<std::string> headerFilename( "h", "header", "Header file", true, "", "Header file" );
	cmd.add( headerFilename );

	
	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	//TCLAP::ValueArg<std::string> inFilenameArg( "i", "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
	//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	std::string inFilename = inFilenameArg.getValue();
	std::string outFilename = outFilenameArg.getValue();
	std::string hdrFilename = headerFilename.getValue();

	std::ifstream headerFile( headerFile );



	headerFile.close();

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
					image = GetSlice<TTYPE>( M4D::Imaging::Image<TTYPE, 3>::Cast( image ), slice ) 
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
