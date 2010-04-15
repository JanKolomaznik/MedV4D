#include <fstream>

#include <QApplication>

#include "common/Common.h"

#include "MainWindow.h"


int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

  QApplication app( argc, argv );
  app.setQuitOnLastWindowClosed( true );

  MainWindow mainWindow;
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
