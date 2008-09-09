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
  if ( mainWindow.wasBuildSuccessful() && mainWindow.wasFilterBuildSuccessful() ) 
  {
    mainWindow.show();
    return app.exec();
  }
  else
  {
    QString message;
    if ( !mainWindow.wasBuildSuccessful() ) {
      message += mainWindow.getBuildMessage() + QString( "\n\n" );
    }
    if ( !mainWindow.wasFilterBuildSuccessful() ) {
      message += mainWindow.getFilterBuildMessage() + QString( "\n\n" );
    }
    message += QObject::tr( "The application will now terminate..." );
    QMessageBox::critical( &mainWindow, QObject::tr( "Exception" ), message );

    return 1;
  } 
}
