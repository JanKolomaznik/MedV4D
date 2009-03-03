#include "backendForDICOM/DICOMServiceProvider.h"
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


	filesystem::path inpath = inFilenameArg.getValue();
	filesystem::path outpath = outFilenameArg.getValue();
	std::string prefix = prefixArg.getValue();


	M4D::Dicom::ResultSet resultSet;

	std::cout << "Searching for DICOM files...\n";
	M4D::Dicom::DcmProvider::LocalFind( resultSet, inpath.string() );

	for( size_t i = 0; i < resultSet.size(); ++i ) {
		M4D::Dicom::TableRow &row = resultSet[i];
		M4D::Dicom::SerieInfoVector info;

		M4D::Dicom::DcmProvider::LocalFindStudyInfo( row.patientID, row.studyID, info );


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
			
			M4D::Dicom::DicomObjSetPtr dcmSet( new M4D::Dicom::DicomObjSet ) ;

			M4D::Dicom::DcmProvider::LocalGetImageSet(
				row.patientID,
				row.studyID,
				info[j].id,
				*dcmSet 
				);

			M4D::Imaging::AbstractImage::Ptr image = 
				M4D::Dicom::DcmProvider::CreateImageFromDICOM( dcmSet );

			M4D::Imaging::ImageFactory::DumpImage( filePath.string(), *image );
			std::cout << "Done\n";
		}
	}
	std::cout << "Conversion finished.\n";
	return 0;
}
