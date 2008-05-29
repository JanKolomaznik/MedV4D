#include <QApplication>

#include "m4dGUIMainWindow.h"

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

  m4dGUIMainWindow mainWindow;
  mainWindow.show();

  return app.exec();
}
