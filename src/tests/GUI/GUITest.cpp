//#include "Imaging/ImageFactory.h"

#include "common/Common.h"

#include <QtGui/QApplication>

#include "GUI/widgets/MainWindow.h"
#include <iostream>
#include <fstream>
#include <sstream>




int
main( int argc, char** argv )
{
        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );
	
	QApplication app(argc, argv);
	M4D::GUI::MainWindow mainWindow;
	mainWindow.show();
	return app.exec();
}

