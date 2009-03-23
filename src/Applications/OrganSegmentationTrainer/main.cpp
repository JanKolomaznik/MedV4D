#include <QApplication>

#include "MainWindow.h"

#include "common/Common.h"
#include <fstream>

#include "Imaging/Imaging.h"
#include "ImageTools.h"

#include <cstdlib>

int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );


	QApplication app( argc, argv );
	app.setQuitOnLastWindowClosed( true );

	MainWindow mainWindow;

	mainWindow.show();
	return app.exec();
}
