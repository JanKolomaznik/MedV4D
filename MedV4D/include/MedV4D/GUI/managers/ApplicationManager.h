#ifndef APPLICATION_MANAGER_H
#define APPLICATION_MANAGER_H

//Temporary workaround
#ifndef Q_MOC_RUN 

#include "MedV4D/GUI/managers/OpenGLManager.h"
#include "MedV4D/GUI/managers/ViewerManager.h"
#include "MedV4D/GUI/managers/DatasetManager.h"

#include "MedV4D/GUI/utils/Module.h"
#include "MedV4D/GUI/utils/Settings.h"

#include "MedV4D/GUI/widgets/MainWindow.h"
#include "MedV4D/GUI/utils/ProxyViewerController.h"
#include "MedV4D/Common/IDGenerator.h"
#include <boost/thread/future.hpp>
#endif

#include <QtCore>

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif



#define GET_SETTINGS( NAME, TYPE, DEFAULT ) \
	ApplicationManager::getInstance()->settings().get<TYPE>( NAME, DEFAULT )
#define SET_SETTINGS( NAME, TYPE, VALUE ) \
	ApplicationManager::getInstance()->settings().set<TYPE>( NAME, VALUE )


class QApplication;

class ApplicationManager: public QObject, public OpenGLManager, public ViewerManager, public DatasetManager
{
	Q_OBJECT
public:
	static ApplicationManager *
	getInstance();

	ApplicationManager();

	virtual void
	initialize( int &argc, char** argv );

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
	addNewMode(ModeViewerController::Ptr aViewerController, M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension )
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
		if( mCurrentMode ) {
			mCurrentMode->viewerController->deactivated();
		}
		mMainWindow->setViewerController( it->second.viewerController );
		mCurrentMode = &(it->second);
		mCurrentMode->viewerController->activated();
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

	QIcon 
	getIcon( QString name )const;
	
	void
	setIconsDirectory( boost::filesystem::path aIconsDirName )
	{
		mIconsDirName = aIconsDirName;
	}
	
	typedef boost::shared_future<void> BackgroundTaskFuture;
	
	BackgroundTaskFuture
	executeBackgroundTask( const boost::function< void () > &aFtor, const QString &aDescription );
	
	template < typename TFunctor >
	BackgroundTaskFuture
	executeBackgroundTask( const TFunctor &aFtor, const QString &aDescription )
	{
		boost::function< void () > func = aFtor;
		return executeBackgroundTask( func, aDescription );
	}
	
	
public slots:
	void
	updateGUIRequest()
	{
		ASSERT( mMainWindow );
		mMainWindow->updateGui();
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
		ModeViewerController::Ptr viewerController;
		M4D::GUI::Viewer::RenderingExtension::Ptr renderingExtension;
	};
	typedef std::map< M4D::Common::IDNumber, ModeInfo > ModeMap;
	ModeMap mModes;
	M4D::Common::IDGenerator mModeIdGenerator;

	void
	viewerSelectionChangedHelper();

	void
	loadIcons();

	bool	mInitialized;
	QApplication *mApp;

	M4D::GUI::MainWindow *mMainWindow;

	ModuleMap	mModules;
	ModeInfo	*mCurrentMode;

	Settings	mSettings;

	boost::filesystem::path mIconsDirName;
	typedef std::map< QString, QIcon > IconMap;
	IconMap mIconMap;

#ifdef USE_TBB
	tbb::task_scheduler_init mTBBScheduler;
#endif /*USE_TBB*/
};

#endif /*APPLICATION_MANAGER_H*/
