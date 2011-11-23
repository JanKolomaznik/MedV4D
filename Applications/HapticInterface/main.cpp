#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>
#include <QApplication>
#include "ViewerWindow.h"

using namespace boost;

int main( int argc, char** argv )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	//////////////////////////////////////////////////////////////////////////
	// MedV4d GUI
	//////////////////////////////////////////////////////////////////////////
	//QApplication app( argc, argv );
	//app.setQuitOnLastWindowClosed( true );

	//MainWindow mainWindow;
	//mainWindow.build();
	//if ( mainWindow.wasBuildSuccessful() ) 
	//{
	//	mainWindow.show();
	//	return app.exec();
	//}
	//else
	//{
	//	QMessageBox::critical( &mainWindow, QObject::tr( "Exception" ), mainWindow.getBuildMessage() + QString( "\n\n" ) +
	//		QObject::tr( "The application will now terminate..." ) );
	//	return 1;
	//} 
	
	////////////////////////////////////////////////////////////////////////
	// Separate Window
	////////////////////////////////////////////////////////////////////////

	if( argc < 2 || argc > 2 ) {
		std::cerr << "Wrong argument count - must be in form: 'program file'\n";
		return 1;
	}

	std::string filename = argv[1];


	std::cout << "Loading file...";
	M4D::Imaging::AImage::Ptr image = 
		M4D::Imaging::ImageFactory::LoadDumpedImage( filename );
	std::cout << "Done\n";
	

	M4D::Imaging::ConnectionTyped< M4D::Imaging::AImage > prodconn;
	prodconn.PutDataset( image );


	QApplication app(argc, argv);
	
	ViewerWindow viewer(prodconn);
	viewer.show();
	return app.exec();

	////////////////////////////////////////////////////////////////////////////
	//// GENERATING TESTING DATA
	////////////////////////////////////////////////////////////////////////////
	//
	//if( argc < 2 || argc > 2 ) {
	//	std::cerr << "Wrong argument count - must be in form: 'program outputfile'\n";
	//	return 1;
	//}

	//std::string fileName = argv[1];

	//unsigned size = 32;
	//unsigned hSize = size / 2;
	//unsigned qSize = hSize / 2;

	//M4D::Imaging::Image< int16, 3 >::Ptr image =
	//M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped< int16 >(size, size, size);

	//float var = 4.0;

	//for( unsigned i=0; i<size; ++i ) {
	//	for( unsigned j=0; j<size; ++j ) {
	//		for (unsigned k=0; k<size; ++k) {
	//			//image->GetElement( i, j ) = ((i>>4)+(j>>4)) & 1 ? 0 : 255;
	//			//image->GetElement( i, j ) = (/*(i>>4)+*/(j)) & 0xf ? 0 : 255;
	//			image->GetElement( Vector< int32, 3 >( i, j, k) ) = (PWR(i-hSize)+PWR(j-hSize)+PWR(k-hSize)) < PWR(qSize) ? (int)((sqrt((float)(PWR(i-hSize)+PWR(j-hSize)+PWR(k-hSize)))) * var) : 0;
	//		}
	//	}
	//}

	//std::cout << "Saving...";
	//M4D::Imaging::ImageFactory::DumpImage( fileName, *image );
	//std::cout << "Done\n";

	//std::cout << "Finished.\n";
	//return 0;
} 

//#include "backendForDICOM/DICOMServiceProvider.h"
//#include "Imaging/ImageFactory.h"
//#include <iostream>
//#include <sstream>
//#include <boost/filesystem/path.hpp>
//#include <tclap/CmdLine.h>
//
//using namespace boost;
//
//int
//main( int argc, char** argv )
//{
//	std::ofstream logFile( "Log.txt" );
//	SET_LOUT( logFile );
//
//	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
//	SET_DOUT( debugFile );
//
//	TCLAP::CmdLine cmd("Tool for conversion DICOM files to dump format.", ' ', "");
//
//	TCLAP::ValueArg<std::string> prefixArg( "p", "prefix", "Prefix of files created from inputs.", false, "DICOMSerie", "Prefix" );
//	cmd.add( prefixArg );
//
//	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
//	//TCLAP::ValueArg<std::string> inFilenameArg( "i", "input", "Input image filename", true, "", "filename1" );
//	cmd.add( inFilenameArg );
//
//	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
//	//TCLAP::ValueArg<std::string> outFilenameArg( "o", "output", "Output image filename", true, "", "filename2" );
//	cmd.add( outFilenameArg );
//
//	cmd.parse( argc, argv );
//
//
//	filesystem::path inpath = inFilenameArg.getValue();
//	filesystem::path outpath = outFilenameArg.getValue();
//	std::string prefix = prefixArg.getValue();
//
//
//	M4D::Dicom::ResultSet resultSet;
//
//	std::cout << "Searching for DICOM files...\n";
//	M4D::Dicom::DcmProvider::LocalFind( resultSet, inpath.string() );
//
//	for( size_t i = 0; i < resultSet.size(); ++i ) {
//		M4D::Dicom::TableRow &row = resultSet[i];
//		M4D::Dicom::SerieInfoVector info;
//
//		M4D::Dicom::DcmProvider::LocalFindStudyInfo( row.patientID, row.studyID, info );
//
//
//		std::cout << "Processing DICOM series...\n";
//		for( size_t j = 0; j < info.size(); ++j ) {
//			std::ostringstream fileNameStream;
//			if( info.size() == 1 ) {
//				fileNameStream << prefix << ".dump";
//			} else {
//				fileNameStream << prefix << j << ".dump";
//			}
//			filesystem::path filePath = outpath;
//			filePath /= fileNameStream.str();
//
//
//			std::cout << "Converting to file '" << fileNameStream.str() << "' ... ";
//
//			M4D::Dicom::DicomObjSetPtr dcmSet( new M4D::Dicom::DicomObjSet ) ;
//
//			M4D::Dicom::DcmProvider::LocalGetImageSet(
//				row.patientID,
//				row.studyID,
//				info[j].id,
//				*dcmSet 
//				);
//
//			M4D::Imaging::AImage::Ptr image = 
//				M4D::Dicom::DcmProvider::CreateImageFromDICOM( dcmSet );
//
//			M4D::Imaging::ImageFactory::DumpImage( filePath.string(), *image );
//			std::cout << "Done\n";
//		}
//	}
//	std::cout << "Conversion finished.\n";
//	return 0;
//} 