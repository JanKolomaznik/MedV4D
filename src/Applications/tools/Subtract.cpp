#include "common/Common.h"
#include "Filtering.h"
#include "Imaging/Image.h"
#undef min
#undef max
#include <tclap/CmdLine.h>


using namespace M4D;
using namespace M4D::Imaging;

template< typename TImage >
void
Subtract( TImage &aImage1, TImage &aImage2 )
{
	if ( aImage1.GetSize() != aImage2.GetSize() ) {
		_THROW_ ErrorHandling::EBadParameter(":hl:");
	}
	typename TImage::Iterator it1 = aImage1.GetIterator();
	typename TImage::Iterator it2 = aImage2.GetIterator();

	while ( it1 != it1.End() &&  it2 != it2.End() ) {
		*it1 = Abs(*it1 - *it2);
		++it1;
		++it2;
	}
}

int
main( int argc, char **argv )
{
	try{  

	TCLAP::CmdLine cmd("Tool for subtracting volumetric DICOM image dumps.", ' ', "");

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "input1" );
	//TCLAP::ValueArg<std::string> inFilenameArg( "i", "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> in2FilenameArg( "input2", "Second input image filename", true, "", "input2" );
	//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
	cmd.add( in2FilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "output" );
	//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	std::string inFilename = inFilenameArg.getValue();
	std::string in2Filename = in2FilenameArg.getValue();
	std::string outFilename = outFilenameArg.getValue();


	std::cout << "Loading file #1..."; std::cout.flush();
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Loading file #2..."; std::cout.flush();
	M4D::Imaging::AImage::Ptr image2 = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( in2Filename );
	std::cout << "Done\n";

	//M4D::Imaging::AImage::Ptr outImage;
	
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		Subtract< IMAGE_TYPE > ( IMAGE_TYPE::Cast( *image ), IMAGE_TYPE::Cast( *image2 ) );
	);

/*	if( firstPoint.isSet() && secondPoint.isSet() ) {
		IMAGE_TYPE_PTR_SWITCH_MACRO( image, 
				image = CropImage< IMAGE_TYPE >( IMAGE, firstPoint.getValue(), secondPoint.getValue() );
				);
	}*/

	std::cout << "Saving file..."; std::cout.flush();
	M4D::Imaging::ImageFactory::DumpImage( outFilename, *image );
	std::cout << "Done\n";

	} catch ( ... ) {
		std::cerr << "error \n";
	}

	return 0;
}

