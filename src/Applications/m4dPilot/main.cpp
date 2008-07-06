#include <QApplication>

#include "mainWindow.h"

#include "Common.h"
#include <fstream>

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "m4dPilot"


int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  QCoreApplication::setOrganizationName( ORGANIZATION_NAME );
  QCoreApplication::setApplicationName( APPLICATION_NAME );

  mainWindow mainWindow;
  mainWindow.show();

  return app.exec();
}
