#include "Imaging/ImageFactory.h"
#include <iostream>
#include <sstream>

#include <QWidget>
#include "MedV4D/GUI/widgets/BasicSliceViewer.h"
#include "Imaging/Imaging.h"
#include "MedV4D/Common/Common.h"
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
	QApplication app(argc, argv);
	try {
		std::cout << "Show window\n";
		ViewerWindow viewer;

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

