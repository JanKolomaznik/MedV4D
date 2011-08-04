#ifndef APPLICATION_MANAGER_H
#define APPLICATION_MANAGER_H


#include "GUI/utils/OpenGLManager.h"
#include "GUI/utils/ViewerManager.h"
#include "GUI/utils/Module.h"
#include "GUI/utils/Settings.h"
#include "GUI/widgets/MainWindow.h"
#include "common/IDGenerator.h"
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

	M4D::Common::IDNumber
	addNewMode( M4D::GUI::Viewer::AViewerController::Ptr aViewerController, M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension )
	{
		M4D::Common::IDNumber id = mModeIdGenerator.NewID();
		ModeInfo mode;
		mode.id = id;
		mode.viewerController = aViewerController;
		mode.renderingExtension = aRenderingExtension;
		mModes[id] = mode;
		mMainWindow->addRenderingExtension( aRenderingExtension );

		return id;
	}

	bool
	activateMode( M4D::Common::IDNumber aId )
	{
		ModeMap::iterator it = mModes.find( aId );
		if ( it == mModes.end() ) {
			LOG( "Mode not found ID = " << aId << "\nmodes available : " << mModes.size() );
			return false;
		}

		mMainWindow->setViewerController( it->second.viewerController );
		LOG( "Mode ID = " << aId << " activated" );
		return true;
	}
		
	void
	createDockWidget( const QString &aName, Qt::DockWidgetArea aArea, QWidget * aWidget )
	{
		ASSERT( mMainWindow );

		mMainWindow->createDockWidget( aName, aArea, aWidget );
	}

	void
	addToolBar( QToolBar * toolbar )
	{
		ASSERT( mMainWindow );

		mMainWindow->addToolBar( toolbar );
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
	struct ModeInfo
	{
		M4D::Common::IDNumber id;
		M4D::GUI::Viewer::AViewerController::Ptr viewerController;
		M4D::GUI::Viewer::RenderingExtension::Ptr renderingExtension;
	};
	typedef std::map< M4D::Common::IDNumber, ModeInfo > ModeMap;
	ModeMap mModes;
	M4D::Common::IDGenerator mModeIdGenerator;

	void
	viewerSelectionChangedHelper();

	bool	mInitialized;
	QApplication *mApp;

	M4D::GUI::MainWindow *mMainWindow;

	ModuleMap	mModules;

	Settings	mSettings;
};

#endif /*APPLICATION_MANAGER_H*/
