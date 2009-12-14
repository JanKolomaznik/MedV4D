#include "common/Common.h"
#include "Imaging/Imaging.h"
#include <boost/filesystem/path.hpp>
#undef min
#undef max
#include <tclap/CmdLine.h>
#include <IL/il.h>


using namespace boost;

template< typename ElementType, unsigned Dim >
void
ConvertTo8Bit( const M4D::Imaging::Image< ElementType, Dim > &inImage, M4D::Imaging::Image< uint8, Dim > &outImage, float zero, float contrast )
{
	ASSERT( inImage.GetSize() == outImage.GetSize() );

	typename M4D::Imaging::Image< ElementType, Dim >::Iterator inIt = inImage.GetIterator();
	typename M4D::Imaging::Image< uint8, Dim >::Iterator outIt = outImage.GetIterator();

	while( ! inIt.IsEnd() ) {
		double tmp = contrast * (-zero + *inIt);
		*outIt = static_cast< uint8 >( ClampToInterval( 0.0, 255.0, tmp ) );
		++inIt;
		++outIt;
	}

}


template< typename ElementType >
void
SaveImageRegion( const M4D::Imaging::ImageRegion< ElementType, 2 > &region, filesystem::path output )
{
	LOG( region.GetElementTypeID() );
	_THROW_ M4D::ErrorHandling::ETODO();
}

/*template<>
void
SaveImageRegion( const M4D::Imaging::ImageRegion< uint8, 2 > &region, filesystem::path output )
{

}*/

template<>
void
SaveImageRegion( const M4D::Imaging::ImageRegion< int16, 2 > &region, filesystem::path output )
{
	ilTexImage( region.GetSize( 0 ), region.GetSize( 1 ), 0, 1, IL_LUMINANCE, IL_SHORT, region.GetPointer() );

	LOG( "IL error1 : " << ilGetError() );

	ilSaveImage( output.file_string().data() ); 

	LOG( "IL error2 : " << ilGetError() );
}

template<>
void
SaveImageRegion( const M4D::Imaging::ImageRegion< uint8, 2 > &region, filesystem::path output )
{
	ilTexImage( region.GetSize( 0 ), region.GetSize( 1 ), 0, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, region.GetPointer() );

	LOG( "IL error1 : " << ilGetError() );

	ilSaveImage( output.file_string().data() ); 

	LOG( "IL error2 : " << ilGetError() );
}

template< typename ElementType >
void
SaveImage( const M4D::Imaging::Image< ElementType, 2 > &image, filesystem::path output )
{
	SaveImageRegion( image.GetRegion(), output );
}

template< typename ElementType >
void
SaveImage( const M4D::Imaging::Image< ElementType, 3 > &image, filesystem::path output )
{
	std::string filename = output.stem();
	std::string extension = output.extension();
	filesystem::path path = output.parent_path();
	
	for( int i = image.GetMinimum()[2]; i < image.GetMaximum()[2]; ++i ) {
		filesystem::path outputNb = path / TO_STRING( filename << (i - image.GetMinimum()[2]) << extension );
		SaveImageRegion( image.GetSlice( i ), outputNb );
	}

}


int
main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );
	
	TCLAP::CmdLine cmd( "Convertor from Medv4D image dump to official picture formats.", ' ', "");
	/*---------------------------------------------------------------------*/

	TCLAP::ValueArg<float> zeroArg( "z", "zero", "Zero position", false, 0, "float" );
	cmd.add( zeroArg );

	TCLAP::ValueArg<float> contrastArg( "c", "contrast", "contrast", false, 1.0, "float" );
	cmd.add( contrastArg );

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image dump filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	/*---------------------------------------------------------------------*/


	std::string inFilename = inFilenameArg.getValue();
	filesystem::path outFilename = outFilenameArg.getValue();
	float zero = zeroArg.getValue();
	float contrast = contrastArg.getValue();
	bool transform = zeroArg.isSet() || contrastArg.isSet();

	std::cout << "Loading file '" << inFilename << "' ..."; std::cout.flush();
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";

	
	//Initialize DevIL
	ilInit();
	ILuint ImgId = 0;
	ilGenImages(1, &ImgId);
	ilBindImage(ImgId);

	if( transform ) {
		M4D::Imaging::AImage::Ptr tmpImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< uint8 >( 
							image->GetDimension(), 
							image->GetMinimumP(), 
							image->GetMaximumP(), 
							image->GetElementExtentsP()
							);

		IMAGE_NUMERIC_TYPE_CONST_REF_SWITCH_MACRO( *image,
			ConvertTo8Bit( IMAGE_TYPE::CastAImage( *image ), M4D::Imaging::Image< uint8, IMAGE_TYPE::Dimension >::CastAImage( *tmpImage ), zero, contrast );
			);
			image = tmpImage;
	}
	
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		SaveImage( IMAGE_TYPE::CastAImage( *image ) , outFilename );
	);


	ilBindImage(0);
	ilDeleteImage(ImgId);
}
