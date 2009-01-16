#include "dicomConn/DICOMServiceProvider.h"
#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>
#include <boost/filesystem/path.hpp>
#include <tclap/CmdLine.h>

using namespace boost;

int
main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	TCLAP::CmdLine cmd("Tool for conversion DICOM files to dump format.", ' ', "");

	TCLAP::ValueArg<std::string> prefixArg( "p", "prefix", "Prefix of files created from inputs.", false, "DICOMSerie", "Slice index" );
	cmd.add( prefixArg );

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	//TCLAP::ValueArg<std::string> inFilenameArg( "i", "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
	//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	M4D::Dicom::DcmProvider dcmProvider;

	filesystem::path inpath = inFilenameArg.getValue();
	filesystem::path outpath = outFilenameArg.getValue();
	std::string prefix = prefixArg.getValue();


	M4D::Dicom::DcmProvider::ResultSet resultSet;

	std::cout << "Searching for DICOM files...\n";
	dcmProvider.LocalFind( resultSet, inpath.string() );

	for( size_t i = 0; i < resultSet.size(); ++i ) {
		M4D::Dicom::DcmProvider::TableRow &row = resultSet[i];
		M4D::Dicom::DcmProvider::SerieInfoVector info;

		dcmProvider.LocalFindStudyInfo( row.patientID, row.studyID, info );


		std::cout << "Processing DICOM series...\n";
		for( size_t j = 0; j < info.size(); ++j ) {
			std::ostringstream fileNameStream;
			if( info.size() == 1 ) {
				fileNameStream << prefix << ".dump";
			} else {
				fileNameStream << prefix << j << ".dump";
			}
			filesystem::path filePath = outpath;
			filePath /= fileNameStream.str();


			std::cout << "Converting to file '" << fileNameStream.str() << "' ... ";
			
			M4D::Dicom::DcmProvider::DicomObjSetPtr dcmSet( new M4D::Dicom::DcmProvider::DicomObjSet ) ;

			dcmProvider.LocalGetImageSet(
				row.patientID,
				row.studyID,
				info[j].id,
				*dcmSet 
				);

			M4D::Imaging::AbstractImage::AImagePtr image = 
				M4D::Imaging::ImageFactory::CreateImageFromDICOM( dcmSet );

			M4D::Imaging::ImageFactory::DumpImage( filePath.string(), *image );
			std::cout << "Done\n";
		}
	}
	std::cout << "Conversion finished.\n";
	return 0;
}
