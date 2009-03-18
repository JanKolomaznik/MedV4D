#include <QApplication>

#include "MainWindow.h"

#include "Common.h"
#include <fstream>

#include "Imaging.h"
#include "ImageTools.h"

#include <cstdlib>

int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	
	//TrainingDataInfos infos;
	//FindTrainingFilePairs( "./TrainingData/", ".idx", infos, false );

	QApplication app( argc, argv );
	app.setQuitOnLastWindowClosed( true );

	MainWindow mainWindow;

	mainWindow.show();
	return app.exec();
}
