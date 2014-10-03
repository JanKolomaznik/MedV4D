#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "MedV4D/DICOMInterface/DcmProvider.h"
#include <QtWidgets>


namespace M4D {

ApplicationManager *appManagerInstance = NULL;

ApplicationManager *
ApplicationManager::getInstance()
{
	return appManagerInstance;
}

ApplicationManager::ApplicationManager()
	: OpenGLManager( static_cast<OpenGLManager*>( this ) ), ViewerManager( static_cast<ViewerManager*>( this ) ), mInitialized( false ), mCurrentMode( NULL )
{
	ASSERT( appManagerInstance == NULL )

	appManagerInstance = this;
}

void
ApplicationManager::initialize( int &argc, char** argv )
{
	Medv4DInit();

	mApp = new QApplication(argc, argv);
	loadIcons();


	M4D::Dicom::DcmProvider::Init();

	OpenGLManager::initialize();
	ViewerManager::initialize();
	DatasetManager::initialize();
	mInitialized = true;
}

void
ApplicationManager::finalize()
{
	ASSERT( mInitialized );

	DatasetManager::finalize();
	ViewerManager::finalize();
	OpenGLManager::finalize();

	M4D::Dicom::DcmProvider::Shutdown();

	mModes.clear();
	mModules.clear();
	mIconMap.clear();
	QApplication::quit();
	delete mApp;
	LOG( "Application manager finalization" );
}

ApplicationManager::~ApplicationManager()
{
	finalize();
}

void
ApplicationManager::setMainWindow( M4D::GUI::MainWindow &aMainWindow )
{
	mMainWindow = &aMainWindow;
}

QIcon
ApplicationManager::getIcon( QString name )const
{
	name = name.toLower();

	IconMap::const_iterator it = mIconMap.find( name );
	if ( it != mIconMap.end() ) {
		return it->second;
	}
	return QIcon();
}

void
ApplicationManager::loadIcons()
{
	//boost::filesystem::path dirName = "./data/icons";
	if (!boost::filesystem::exists(mIconsDirName)) {
		LOG( "Directory \'" << mIconsDirName << "\' doesn't exist!" );
		return;
	}
	if (!boost::filesystem::is_directory(mIconsDirName) ){
		LOG( "\'" << mIconsDirName << "\' is not a directory!" );
		return;
	}

	boost::filesystem::directory_iterator dirIt(mIconsDirName);
	boost::filesystem::directory_iterator end;
	for ( ;dirIt != end; ++dirIt ) {
		D_PRINT( "Found file :" << *dirIt );
		boost::filesystem::path p = dirIt->path();
		if ( p.extension() == ".png" || p.extension() == ".svg" ) {
			QString filename = p.string().c_str();
			QString name = p.stem().string().c_str();
			if ( !filename.isEmpty() && !name.isEmpty() ) {
				mIconMap[ name.toLower() ] = QIcon( filename );
			}
		}
	}

	/*"2d"
	"3d"
	"jittering"
	"shading"
	"bounding_box"
	"cut_plane"
	"volume_restrictions"
	"XY"
	"YZ"
	"XZ"
	"reset_view"
	"low_quality"
	"normal_quality"
	"high_quality"
	"finest_quality"*/
}


ApplicationManager::BackgroundTaskFuture
ApplicationManager::executeBackgroundTask( const boost::function< void () > &aFtor, const QString &aDescription )
{
	boost::packaged_task<void> task( aFtor );
	boost::unique_future<void> uniqueFut = task.get_future();
	BackgroundTaskFuture fut = boost::move( uniqueFut );

	boost::thread backThread( boost::move( task ) ); // launch task on a thread
	//TODO better GUI response


	return fut;
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
	//D_PRINT( "Viewer selection changed" );
	emit viewerSelectionChanged();
}

void
ApplicationManager::notifyAboutChangedViewerSettings()
{
	emit selectedViewerSettingsChanged();
}


} //namespace M4D
