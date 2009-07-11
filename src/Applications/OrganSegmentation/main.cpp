#include <QApplication>

#include "mainWindow.h"

#include "common/Common.h"
#include <fstream>

#include "MainManager.h"
#include "ManualSegmentationManager.h"
#include "KidneySegmentationManager.h"
#include "SegmentationManager.h"

#include "Imaging/Histogram.h"

#include <cstdlib>

int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	QApplication app( argc, argv );
	app.setQuitOnLastWindowClosed( true );

	MainManager::Instance().Initialize();
	
	InitializeSegmentationManagers();

	mainWindow mainWindow;
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
