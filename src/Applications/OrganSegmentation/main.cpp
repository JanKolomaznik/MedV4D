#include <QApplication>

#include "mainWindow.h"

#include "Common.h"
#include <fstream>

#include "MainManager.h"
#include "ManualSegmentationManager.h"
#include "KidneySegmentationManager.h"

#include "Imaging/Histogram.h"

#include <cstdlib>

int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	/*M4D::Imaging::Histogram< int32 > histogram( 20, 40, false );
	for( unsigned i = 0; i < 60000; ++i ) {
		histogram.IncCell( rand() % 30 + 15 );
	}
	std::cout << histogram  << "\n";
	std::cerr << histogram.GetSum() << "\n";*/   
	

	MainManager::Instance().Initialize();
	ManualSegmentationManager::Instance().Initialize();
	KidneySegmentationManager::Instance().Initialize();

	QApplication app( argc, argv );
	app.setQuitOnLastWindowClosed( true );

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
