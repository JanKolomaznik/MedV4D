#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "GUI/widgets/BasicSliceViewer.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"
#include "ViewerWindow.hpp"

#include <tclap/CmdLine.h>

std::string inFilename;

void
processCommandLine( int argc, char** argv )
{
	TCLAP::CmdLine cmd( "Median filter.", ' ', "");
	/*---------------------------------------------------------------------*/

	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", false, "", "filename" );
	cmd.add( inFilenameArg );

	cmd.parse( argc, argv );

	inFilename = inFilenameArg.getValue();
}

int
main( int argc, char** argv )
{
	//std::ofstream logFile( "Log.txt" );
        //SET_LOUT( logFile );

        //D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        //SET_DOUT( debugFile );


	QApplication app(argc, argv);
	try {
		processCommandLine( argc, argv );

		std::cout << "Show window\n";
		//ViewerWindow viewer( prodconn );
		ViewerWindow viewer;


		viewer.show();
		if ( !inFilename.empty() ) {
			viewer.openFile( QString::fromStdString( inFilename ) );
		}
		//viewer.applyTransferFunction();
		return app.exec();
	} catch ( std::exception &e )
	{
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	} 
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Unknown error" );
	}
	
	return 1;
}

