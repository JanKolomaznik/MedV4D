#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "GUI/widgets/BasicSliceViewer.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"
#include "ViewerWindow.hpp"


#include "GUI/utils/ApplicationManager.h"

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

#include "AnnotationModule/AnnotationModule.hpp"
#include "ShoulderMeasurementModule/ShoulderMeasurementModule.hpp"
void
createModules()
{
	ApplicationManager *appManager = ApplicationManager::getInstance();
	
	//appManager->addModule( createModule< AnnotationModule >() );
	appManager->addModule( createModule< ShoulderMeasurementModule >() );
}

int
main( int argc, char** argv )
{
	//std::ofstream logFile( "Log.txt" );
        //SET_LOUT( logFile );

        //D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        //SET_DOUT( debugFile );


	ApplicationManager appManager;

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

