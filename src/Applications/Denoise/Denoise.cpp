#pragma warning(disable:4100)

#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>
#include <tclap/CmdLine.h>

#include <QWidget>
#include "GUI/widgets/m4dGUISliceViewerWidget.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"

#define ONLY_VIEWER

#include "progress.h"
#include "globals.h"
#include "volumeset.h"

#define LOG_ERR(args) LOG("Error: " << args)

typedef M4D::Imaging::Image<uint16, 3> TImage16x3;

int
main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	std::string srcfilename;
	std::string dstfilename;

	try {  
		TCLAP::CmdLine cmd("Tool for denoising DICOM files", ' ', "0.1");

		TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
		//TCLAP::ValueArg<std::string> inFilenameArg( "i", "input", "Input image filename", true, "", "filename1" );
		cmd.add( inFilenameArg );

		TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
		//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
		cmd.add( outFilenameArg );

		cmd.parse( argc, argv );

		srcfilename = inFilenameArg.getValue();
		dstfilename = outFilenameArg.getValue();
	} catch(TCLAP::ArgException &e) {
		LOG("error: " << e.error() << " for arg " << e.argId() << std::endl);
	}

	M4D::Imaging::AImage::Ptr image;
	std::cout << "Loading file... ";
	try {
		image = M4D::Imaging::ImageFactory::LoadDumpedImage( srcfilename );
	} catch(M4D::ErrorHandling::ExceptionBase&) {
		std::cout  << "Failed\n";
		return 1;
	}
	std::cout << "Done\n";

	TImage16x3::Ptr myImg = TImage16x3::Cast(image);

	std::cout << "Getting info about file...\n";
	TImage16x3::SizeType size;
	TImage16x3::PointType stride;
	uint16 *pData = myImg->GetPointer(size, stride); // ptr na uint16

	std::cout << "Creating volume...\n";

	viewer::CVolumeSet<uint16> vol16(size[0],size[1],size[2]);
	for(unsigned int z = 0; z < size[2]; z++)
		memcpy(vol16.getXYPlaneNonconst(z), pData + stride[2] * z, stride[2] * sizeof(uint16));

	std::cout << "Converting data...\n";

	viewer::CVolumeSet<float> volF(1,1,1), volFRes(1,1,1);
	volF.copyVolume(vol16);

	std::cout << "Denoising...\n";

	viewer::CTextProgress progress;
	volFRes.volBlockwiseNLMeans(volF, 0.9, 0.1, 0.5, 5, 2, 4, &progress);

	std::cout << "Processing result...\n";

	vol16.copyVolume(volFRes);

	for(unsigned int z = 0; z < size[2]; z++)
		memcpy(pData + stride[2] * z, vol16.getXYPlane(z), stride[2] * sizeof(uint16));

	std::cout << "Saving to disk...\n";

	M4D::Imaging::ImageFactory::DumpImage(dstfilename, *myImg);

	return 0;
}

