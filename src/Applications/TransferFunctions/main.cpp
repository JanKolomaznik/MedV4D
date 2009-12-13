#include <QApplication>

#include "mainWindow.h"

#include "common/Common.h"
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
  mainWindow.build("TransferFunctions","MFF");
  if ( mainWindow.wasBuildSuccessful() ) 
  {
    mainWindow.show();
    return app.exec();
  }
  else
  {
    QMessageBox::critical( &mainWindow, QObject::tr( "Exception" ), mainWindow.getBuildMessage() + QString( "\n\n" ) +
                           QObject::tr( "The application will now terminate..." ) );
    return 1;
  } 
}
