#include "MedV4D/Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "MedV4D/GUI/widgets/BasicSliceViewer.h"
#include "MedV4D/Imaging/Imaging.h"
#include "MedV4D/Common/Common.h"
#include "ViewerWindow.hpp"


#include "MedV4D/GUI/utils/ApplicationManager.h"

#include <tclap/CmdLine.h>

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
#include "AnnotationModule/AnnotationModule.hpp"
#include "ShoulderMeasurementModule/ShoulderMeasurementModule.hpp"
#endif
void
createModules()
{
	ApplicationManager *appManager = ApplicationManager::getInstance();

#ifdef EXTENSION_MODULES_ENABLED	
	appManager->addModule( createModule< AnnotationModule >() );
	appManager->addModule( createModule< ShoulderMeasurementModule >() );
#endif
}

int
main( int argc, char** argv )
{
	//std::ofstream logFile( "Log.txt" );
        //SET_LOUT( logFile );

        //D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        //SET_DOUT( debugFile );

	ApplicationManager appManager;

	boost::filesystem::path dirName = GET_SETTINGS( "gui.icons_directory", std::string, std::string( "./data/icons" ) );
	appManager.setIconsDirectory(dirName);
	appManager.initialize( argc, argv );

	try {
		//processCommandLine( argc, argv );
		ViewerWindow viewer;
		appManager.setMainWindow( viewer );

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


