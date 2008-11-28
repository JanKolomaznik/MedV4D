#include "dicomConn/DICOMServiceProvider.h"
#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

int
main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	if( argc < 3 || argc > 3 ) {
		std::cerr << "Wrong argument count - must be in form: 'program inputdir outputdir'\n";
		return 1;
	}

	M4D::Dicom::DcmProvider dcmProvider;
	std::string inpath = argv[1];
	std::string outpath = argv[2];


	M4D::Dicom::DcmProvider::ResultSet resultSet;

	std::cout << "Searching for DICOM files...\n";
	dcmProvider.LocalFind( resultSet, inpath );

	for( size_t i = 0; i < resultSet.size(); ++i ) {
		M4D::Dicom::DcmProvider::TableRow &row = resultSet[i];
		M4D::Dicom::DcmProvider::SerieInfoVector info;

		dcmProvider.LocalFindStudyInfo( row.patientID, row.studyID, info );


		std::cout << "Processing DICOM series...\n";
		for( size_t j = 0; j < info.size(); ++j ) {
			std::ostringstream fileNameStream;
			fileNameStream << outpath << "/" << "DICOMSerie_" << j << ".dump";
			std::string fileName = fileNameStream.str();


			std::cout << "Converting to file '" << fileName << "' ... ";
			
			M4D::Dicom::DcmProvider::DicomObjSetPtr dcmSet( new M4D::Dicom::DcmProvider::DicomObjSet ) ;

			dcmProvider.LocalGetImageSet(
				row.patientID,
				row.studyID,
				info[j].id,
				*dcmSet 
				);

			M4D::Imaging::AbstractImage::AImagePtr image = 
				M4D::Imaging::ImageFactory::CreateImageFromDICOM( dcmSet );

			M4D::Imaging::ImageFactory::DumpImage( fileName, *image );
			std::cout << "Done\n";
		}
	}
	std::cout << "Conversion finished.\n";
	return 0;
}
