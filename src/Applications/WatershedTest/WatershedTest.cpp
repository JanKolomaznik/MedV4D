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
#include "timer.h"
#include "volumeset.h"

#include <list>
#include <iostream>

#define LOG_ERR(args) LOG("Error: " << args)

typedef M4D::Imaging::Image<uint16, 3> TImage16x3;


class IntMinMaxConstraint : public TCLAP::Constraint<int> {
	int iMin, iMax;
public:
	IntMinMaxConstraint(int iMinimum, int iMaximum) : iMin(iMinimum), iMax(iMaximum) {}
	virtual std::string description() const { 
		std::stringstream ss;
		ss << "must be in interval [" << iMin << ", " << iMax << "]";
		return ss.str();; 
	}
	virtual std::string shortID() const {return "number"; }
	virtual bool check(const int& value) const {
		if(value < iMin || value > iMax)
			return false;
		return true;
	}
};

class IntMinConstraint : public TCLAP::Constraint<int> {
	int iMin;
public:
	IntMinConstraint(int iMinimum) : iMin(iMinimum) {}
	virtual std::string description() const { 
		std::stringstream ss;
		ss << "must be greater or equal to " << iMin;
		return ss.str();; 
	}
	virtual std::string shortID() const {return "ID"; }
	virtual bool check(const int& value) const {
		if(value < iMin)
			return false;
		return true;
	}
};

int
main( int argc, char** argv )
{
//	std::ofstream logFile( "Log.txt" );
//        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	std::string srcfilename;
	std::string dstfilename;

	bool bVerbose;

	try {  
		TCLAP::CmdLine cmd("Watershed segmentation test for Medv4D DICOM dump files", ' ', "0.1");
		
//		MyOutput myOutput;
//		cmd.setOutput(&myOutput);
		cmd.setExceptionHandling(false);

		IntMinMaxConstraint csrRadius(1, 4);
		IntMinMaxConstraint csrNbh(1, 2);
		IntMinConstraint csrIds(0);
		TCLAP::ValueArg<int> argPlatform("p", "platform", "Platform ID (see -l or -a for IDs)", false, 0, (TCLAP::Constraint<int> *) &csrIds, cmd);
		TCLAP::ValueArg<int> argDevice("d", "device", "Device ID (see -l or -a for IDs)", false, 0, (TCLAP::Constraint<int> *) &csrIds, cmd);
		TCLAP::SwitchArg argVerbose("v", "verbose", "Print parameters", cmd);


		TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "source", "Input image filename", true, "", "source" );
		std::vector<TCLAP::Arg*> xorlist;
		xorlist.push_back(&inFilenameArg);
		cmd.xorAdd( xorlist );

		TCLAP::ValueArg<std::string> outFilenameArg("o", "output", "Optional output image filename", false, "", "output" );
		cmd.add( outFilenameArg );

		cmd.parse( argc, argv );

		srcfilename = inFilenameArg.getValue();
		bVerbose = argVerbose.getValue();

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
			ss << srcfilename.substr(0, nameLength) << "_watershed" << srcfilename.substr(nameLength, totalLength - nameLength);
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

	if(bVerbose) {
		std::cout << "Running with parameters:" << std::endl;
		std::cout << "\tSource:\t\t" << srcfilename << std::endl;
		std::cout << "\tDestination:\t" << dstfilename << std::endl;
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
	
	std::cout << "Size of volume: (" << size[0] << ", " << size[1] << ", " << size[2] << ")\n";

	std::cout << "Creating volume...\n";

	viewer::CVolumeSet<uint16> vol16(size[0],size[1],size[2]);
	for(unsigned int z = 0; z < size[2]; z++)
		memcpy(vol16.getXYPlaneNonconst(z), pData + stride[2] * z, stride[2] * sizeof(uint16));

	std::cout << "Converting data...\n";

	viewer::CVolumeSet<float> volF(1,1,1), volFiltered(1,1,1), volFGrad(1,1,1), volFRes(1,1,1);
	volF.copyVolume(vol16);
	volFiltered.volMedianFilter(volF, 3, NULL);

	std::cout << "Computing gradient...\n";	
	viewer::volGradientSizeApprox<float>(volFGrad, volFiltered, 2);
	//volFGrad.copyVolume(volF);
	//viewer::volCreateTestData<float>(volF); // HACK;
	//viewer::volCreateTestData<float>(volFGrad); // HACK;

	std::cout << "Computing local minima...\n";
	viewer::CVolumeSet<int> volMarkers(1,1,1);
	viewer::volGetLocalMinima(volFGrad, volMarkers);

	std::cout << "Segmenting...\n";

	viewer::CTextProgress progress;
	//volF.copyVolume(volFGrad);
	bool retval = viewer::volWatershedBasic<float>(volFRes, volFGrad, volF, volMarkers);

	if (retval == false) {
		std::cout << "Errors encountered.\n";
		return 1;
	}

	std::cout << "Processing result...\n";

	vol16.copyVolume(volFRes);

	for(unsigned int z = 0; z < size[2]; z++)
		memcpy(pData + stride[2] * z, vol16.getXYPlane(z), stride[2] * sizeof(uint16));

	std::cout << "Saving to disk...\n";

	M4D::Imaging::ImageFactory::DumpImage(dstfilename, *myImg);

	std::cout << "Finished.\n";

	return 0;
}

