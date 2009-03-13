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
	//std::cout << M4D::Imaging::Geometry::PointLineSegmentDistanceSquared( Vector< float32, 2 >( 4.0f, 8.0f ), Vector< float32, 2 >( 3.0f, 5.0f ), Vector< float32, 2 >( 4.0f, 0.0f ) ) << "\n";
	//std::cout << M4D::Imaging::Geometry::PointLineSegmentDistanceSquared( Vector< float32, 2 >( 4.0f, 3.0f ), Vector< float32, 2 >( 3.0f, 5.0f ), Vector< float32, 2 >( 4.0f, 0.0f ) ) << "\n";
	

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
