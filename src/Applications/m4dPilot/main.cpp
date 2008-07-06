#include <QApplication>

#include "mainWindow.h"

#include "Common.h"
#include <fstream>


int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  mainWindow mainWindow;
  mainWindow.show();

  return app.exec();
}
