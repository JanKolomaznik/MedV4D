#include "MedV4D/Common/Common.h"
#include "Filtering.h"
#include "Imaging/Image.h"
#undef min
#undef max
#include <tclap/CmdLine.h>
	
#include <fstream>
#include <iostream>

int
main( int argc, char **argv )
{
	try{  

	TCLAP::CmdLine cmd("Tool for croping multidimensional images.", ' ', "");

	TCLAP::ValueArg<std::string> headerFilename( "i", "info", "Header file", true, "", "Header file" );
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

	std::ifstream headerFile( hdrFilename.data(), std::ifstream::in );

	Vector3u size;
	Vector3f voxelSize;
	unsigned bitCount;

	headerFile >> size[0] >> size[1] >> size[2];
	headerFile >> bitCount;
	headerFile >> voxelSize[0] >> voxelSize[1] >> voxelSize[2];

	LOG( "size : " << size );
	LOG( "bits : " << bitCount );
	LOG( "voxelSize : " << voxelSize );
	headerFile.close();

	M4D::Imaging::AImage::Ptr aimage;
	switch ( bitCount ) {
	case 8:{
			M4D::Imaging::Image< uint8, 3 >::Ptr image =
				M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< uint8, 3 >( Vector3u(), size, voxelSize );
			M4D::Imaging::ImageFactory::LoadRawDump( inFilename, *image );
			aimage = image;
	       }

		break;
	case 16:{
			M4D::Imaging::Image< uint16, 3 >::Ptr image =
				M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< uint16, 3 >( Vector3u(), size, voxelSize );
			M4D::Imaging::ImageFactory::LoadRawDump( inFilename, *image );
			aimage = image;
	       }
		break;
	default:
		ASSERT( false );
	}


	std::cout << "Saving file..."; std::cout.flush();
	M4D::Imaging::ImageFactory::DumpImage( outFilename, *aimage );
	std::cout << "Done\n";

	} catch( std::exception &e ) {
		std::cerr << e.what() << std::endl;
	}	
	catch ( ... ) {
		std::cerr << "error \n";
	}

	return 0;
}
