#include "MedV4D/Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "MedV4D/Imaging/Imaging.h"
#include "MedV4D/Common/Common.h"
#include "ViewerWindow.hpp"


#include "MedV4D/GUI/managers/ApplicationManager.h"

#include <tclap/CmdLine.h>

//#include <X11/Xlib.h>

std::string inFilename;

void
processCommandLine( int argc, char** argv )
{
	TCLAP::CmdLine cmd( "Viewer", ' ', "");
	/*---------------------------------------------------------------------*/

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", false, "", "filename" );
	cmd.add( inFilenameArg );

	cmd.parse( argc, argv );

	inFilename = inFilenameArg.getValue();
}
#ifdef EXTENSION_MODULES_ENABLED
//#include "AnnotationModule/AnnotationModule.hpp"
#include "ShoulderMeasurementModule/ShoulderMeasurementModule.hpp"
//#include "OrganSegmentationModule/OrganSegmentationModule.hpp"
#endif
void
createModules()
{
#ifdef EXTENSION_MODULES_ENABLED
	ApplicationManager *appManager = ApplicationManager::getInstance();
//	appManager->addModule( createModule< AnnotationModule >() );
	appManager->addModule( createModule< ShoulderMeasurementModule >() );
//	appManager->addModule( createModule< OrganSegmentationModule >() );
#endif
}

int
main( int argc, char** argv )
{
	//std::ofstream logFile( "Log.txt" );
	//SET_LOUT( logFile );

	//D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	//SET_DOUT( debugFile );

	//XInitThreads();
	ApplicationManager appManager;

	boost::filesystem::path executablePath(argv[0]);
	boost::filesystem::path dataDirName = GET_SETTINGS( "application.data_directory", std::string, (boost::filesystem::path(argv[0]).parent_path() / "data").string() );
	//If we cannot locate data directory - try other posiible locations
	if (!boost::filesystem::exists(dataDirName) || !boost::filesystem::is_directory(dataDirName)) {
		std::vector<boost::filesystem::path> possibleDataDirs;
		possibleDataDirs.push_back(boost::filesystem::current_path() / "data");
		possibleDataDirs.push_back(executablePath.parent_path() / "data");
		possibleDataDirs.push_back(executablePath.parent_path().parent_path() / "data");

		std::vector<boost::filesystem::path>::const_iterator it = possibleDataDirs.begin();
		bool found = false;
		LOG( "Trying to locate 'data' directory:" );
		while (!found && it != possibleDataDirs.end()) {
			LOG_CONT( "\tChecking: " << it->string() << " ... ");
			if (boost::filesystem::exists(*it) && boost::filesystem::is_directory(*it)) {
				dataDirName = *it;
				SET_SETTINGS( "application.data_directory", std::string, dataDirName.string() );
				found = true;
				LOG( "SUCCESS" );
			} else {
				LOG( "FAILED" );
			}
			++it;
		}
		if (!found) {
			BOOST_THROW_EXCEPTION( M4D::ErrorHandling::EDirNotFound() );
		}
	}
	boost::filesystem::path dirName = GET_SETTINGS( "gui.icons_directory", std::string, ( dataDirName / "icons" ).string() );
	appManager.setIconsDirectory(dirName);
	appManager.initialize( argc, argv );

	try {
		//processCommandLine( argc, argv );
		ViewerWindow viewer;
		appManager.setMainWindow( viewer );
		viewer.initialize();

		createModules();

		appManager.loadModules();
		viewer.showMaximized();
		return appManager.exec();
	} catch ( std::exception &e )
	{
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Unknown error" );
	}

	return 1;
}


