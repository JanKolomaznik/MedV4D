#include "GUI/utils/ApplicationManager.h"
#include <QtGui>


ApplicationManager *appManagerInstance = NULL;

ApplicationManager *
ApplicationManager::getInstance()
{
	return appManagerInstance;
}

ApplicationManager::ApplicationManager()
	: OpenGLManager( static_cast<OpenGLManager*>( this ) ), mInitialized( false )
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
	mInitialized = true;
}

void
ApplicationManager::finalize()
{
	ASSERT( mInitialized );

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

