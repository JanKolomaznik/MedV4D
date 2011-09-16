#pragma warning(disable:4100)

#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>
#include <tclap/CmdLine.h>

//#include <QWidget>
//#include "GUI/widgets/m4dGUISliceViewerWidget.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"

#define ONLY_VIEWER
#define OPENCL

#include "progress.h"
#include "globals.h"
#include "timer.h"
#include "volumeset.h"

#include <list>
#include <iostream>

#define LOG_ERR(args) LOG("Error: " << args)

typedef M4D::Imaging::Image<uint16, 3> TImage16x3;


int
main( int argc, char** argv )
{
//	std::ofstream logFile( "Log.txt" );
//        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	std::string srcfilename;
	std::string dstfilename;

	bool bVolumeset;
	bool bDump;

	try {  
		TCLAP::CmdLine cmd("Tool for converting between DicomView volumeset and MedV4D dicomdump datasets.", ' ', "0.1");
		
//		MyOutput myOutput;
//		cmd.setOutput(&myOutput);
		cmd.setExceptionHandling(false);

		TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "source", "Input image filename", true, "", "source", cmd );

		TCLAP::ValueArg<std::string> outFilenameArg("o", "output", "Optional output image filename", false, "", "output" );
		cmd.add( outFilenameArg );

		TCLAP::SwitchArg argVolumeset("v", "volumeset", "Output is volumeset");
		TCLAP::SwitchArg argDump("d", "dump", "Output is MedV4D dump");
		std::vector<TCLAP::Arg*> xorlist;
		xorlist.push_back(&argVolumeset);
		xorlist.push_back(&argDump);
		cmd.xorAdd( xorlist );

		cmd.parse( argc, argv );

		srcfilename = inFilenameArg.getValue();
		bVolumeset = argVolumeset.getValue();
		bDump = argDump.getValue();

		if(bVolumeset == false && bDump == false) {
			throw(TCLAP::CmdLineParseException("Output type not specified!"));
		}

		if(outFilenameArg.isSet()) {
			dstfilename = outFilenameArg.getValue();
		} else {
			int nameLength, totalLength;
			const char* szFilename = srcfilename.c_str();
			const char* pDot = strrchr(szFilename, '.');
			totalLength = (int)srcfilename.length();
			if(pDot != NULL)
				nameLength = (int)((pDot - szFilename) / sizeof(char));
			else
				nameLength = totalLength;

			std::stringstream ss;
			ss << srcfilename.substr(0, nameLength);// << srcfilename.substr(nameLength, totalLength - nameLength);
			ss << ((bVolumeset)?".vol":".dump");
			dstfilename = ss.str();
		}

	} catch(TCLAP::ArgException &e) {
		if(e.argId() != " ") {
			LOG("Error: " << e.error() << " for " << e.argId());
		} else {
			LOG("Error: " << e.error());
		}
		LOG("Use --help for more info.");
		return 1;
	} catch(TCLAP::ExitException &e) {
		return e.getExitStatus();
	}

	// dump -> volumeset
	if(bVolumeset) {
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
		std::cout << "Saving to disk...\n";
		volF.saveToDisk(dstfilename.c_str(), NULL);
		std::cout << "Finished.\n";

	} else {	// volumeset->dump

		viewer::CVolumeSet<float> volF(1,1,1);
		std::cout << "Loading file... ";
		if(false == volF.loadFromDisk(srcfilename.c_str(), NULL)) {
			std::cout  << "Failed\n";
			return 1;
		}
		std::cout << "Done\n";

		std::cout << "Converting data...\n";

		viewer::CVolumeSet<uint16> vol16(1,1,1);
		vol16.copyVolume(volF);

		TImage16x3::Ptr image = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents<uint16>( 
			Vector< int32, 3 > (), 
			Vector< int32, 3 > (vol16.getWidth(), vol16.getHeight(), vol16.getDepth()),
			Vector< float32, 3 > (1.0f, 1.0f, 1.0f) );

		TImage16x3::SizeType size;
		TImage16x3::PointType stride;
		uint16 *pData = image->GetPointer(size, stride); // ptr na uint16

		for(unsigned int z = 0; z < size[2]; z++)
			memcpy(pData + stride[2] * z, vol16.getXYPlaneNonconst(z), stride[2] * sizeof(uint16));

		std::cout << "Saving to disk...\n";

		M4D::Imaging::ImageFactory::DumpImage(dstfilename, *image);
	}
/*	std::cout << "Processing result...\n";

	vol16.copyVolume(volFRes);

	for(unsigned int z = 0; z < size[2]; z++)
		memcpy(pData + stride[2] * z, vol16.getXYPlane(z), stride[2] * sizeof(uint16));

	std::cout << "Saving to disk...\n";

	M4D::Imaging::ImageFactory::DumpImage(dstfilename, *myImg);*/

	std::cout << "Finished.\n";

	return 0;
}

