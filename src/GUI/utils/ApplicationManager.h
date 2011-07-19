#ifndef APPLICATION_MANAGER_H
#define APPLICATION_MANAGER_H


#include "GUI/utils/OpenGLManager.h"
#include "GUI/utils/ViewerManager.h"
#include "GUI/utils/Module.h"
#include "GUI/utils/Settings.h"
#include "GUI/widgets/MainWindow.h"
#include <QtCore>


#define GET_SETTINGS( NAME, TYPE, DEFAULT ) \
	ApplicationManager::getInstance()->settings().get<TYPE>( NAME, DEFAULT )


class QApplication;

class ApplicationManager: public QObject, public OpenGLManager, public ViewerManager
{
	Q_OBJECT
public:
	static ApplicationManager *
	getInstance();

	ApplicationManager();

	virtual void
	initialize( int argc, char** argv );

	virtual void
	finalize();

	virtual int
	exec();

	~ApplicationManager();

	void
	notifyAboutChangedViewerSettings();

	void
	addModule( AModule::Ptr aModule )
	{
		if( aModule ) {
			ModuleMap::iterator it = mModules.find( aModule->getName() );
			if( it == mModules.end() ) {
				mModules[ aModule->getName() ] = aModule;
			} else {
				_THROW_ M4D::ErrorHandling::EAlreadyPresent();
			}
		} else {
			_THROW_ M4D::ErrorHandling::ENULLPointer();
		}
	}

	void
	loadModules()
	{
		ModuleMap::iterator it = mModules.begin();
		for( ; it != mModules.end(); ++it ) {
			if ( ! it->second->isLoaded() ) {
				it->second->load();
			}
		}
	}

	void
	setMainWindow( M4D::GUI::MainWindow &aMainWindow )
	{
		mMainWindow = &aMainWindow;
	}

	Settings &
	settings()
	{
		return mSettings;
	}
signals:
	void
	viewerSelectionChanged();

	void
	selectedViewerSettingsChanged();
protected:
	void
	viewerSelectionChangedHelper();

	bool	mInitialized;
	QApplication *mApp;

	M4D::GUI::MainWindow *mMainWindow;

	ModuleMap	mModules;

	Settings	mSettings;
};

#endif /*APPLICATION_MANAGER_H*/
