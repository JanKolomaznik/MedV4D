#include "GUI/utils/ApplicationManager.h"
#include <QtGui>


ApplicationManager *appManagerInstance = NULL;

ApplicationManager *
ApplicationManager::getInstance()
{
	return appManagerInstance;
}

ApplicationManager::ApplicationManager()
	: OpenGLManager( static_cast<OpenGLManager*>( this ) ), ViewerManager( static_cast<ViewerManager*>( this ) ), mInitialized( false )
{
	ASSERT( appManagerInstance == NULL )

	appManagerInstance = this;
}

void
ApplicationManager::initialize( int argc, char** argv )
{
	Medv4DInit();
	mApp = new QApplication(argc, argv);

	OpenGLManager::initialize();
	ViewerManager::initialize();
	mInitialized = true;
}

void
ApplicationManager::finalize()
{
	ASSERT( mInitialized );

	ViewerManager::finalize();
	OpenGLManager::finalize();
	delete mApp;
}

ApplicationManager::~ApplicationManager()
{
	finalize();
}

int
ApplicationManager::exec()
{
	ASSERT( mInitialized );
	return mApp->exec();
}

void
ApplicationManager::viewerSelectionChangedHelper()
{
	D_PRINT( "Viewer selection changed" );
	emit viewerSelectionChanged();
}

