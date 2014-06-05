#include "ViewerWindow.hpp"
//#include "MedV4D/GUI/utils/ImageDataRenderer.h"
#include "MedV4D/Imaging/ImageTools.h"
#include "MedV4D/Imaging/Histogram.h"
#include <cmath>

#include "MedV4D/GUI/widgets/PythonTerminal.h"
#include "MedV4D/GUI/widgets/MultiDockWidget.h"
#include "MedV4D/GUI/managers/ViewerManager.h"
#include "MedV4D/GUI/utils/ProxyRenderingExtension.h"
#include "MedV4D/GUI/utils/ViewerAction.h"
#include "MedV4D/GUI/utils/QtM4DTools.h"
#include "MedV4D/GUI/widgets/SettingsDialog.h"
#include "MedV4D/GUI/widgets/ViewerControls.h"
#include "MedV4D/Common/MathTools.h"
#include <boost/thread.hpp>

#include <iostream>
#include <iterator>
#include <algorithm>

#ifdef USE_CUDA
#include "MedV4D/Imaging/cuda/MedianFilter.h"
#endif

#include <QtGui>
namespace M4D
{
namespace GUI
{
namespace Viewer
{

class GeneralViewerFactory: public AViewerFactory
{
public:
	GeneralViewerFactory() :mPrimaryConnection( NULL ), mSecondaryConnection( NULL )
	{}
	typedef std::shared_ptr< GeneralViewerFactory > Ptr;

	AGLViewer *
	createViewer()
	{
		GeneralViewer *viewer = new GeneralViewer();
		if( mRenderingExtension ) {
			viewer->setRenderingExtension( mRenderingExtension );
		}
		if( mViewerController ) {
			viewer->setViewerController( mViewerController );
		}
		if( mPrimaryConnection ) {
			mPrimaryConnection->ConnectConsumer( viewer->InputPort()[0] );
		}
		if( mSecondaryConnection ) {
			mSecondaryConnection->ConnectConsumer( viewer->InputPort()[1] );
		}

		viewer->setLUTWindow(glm::fvec2(1500.0f,100.0f));

		viewer->enableShading( GET_SETTINGS( "gui.viewer.volume_rendering.shading_enabled", bool, true ) );

		viewer->enableJittering( GET_SETTINGS( "gui.viewer.volume_rendering.jittering_enabled", bool, true ) );

		viewer->setJitterStrength( GET_SETTINGS( "gui.viewer.volume_rendering.jitter_strength", double, 1.0 ) );

		viewer->enableIntegratedTransferFunction( GET_SETTINGS( "gui.viewer.volume_rendering.integrated_transfer_function_enabled", bool, true ) );

		viewer->setRenderingQuality( GET_SETTINGS( "gui.viewer.volume_rendering.rendering_quality", int, qmNormal ) );

		viewer->enableBoundingBox( GET_SETTINGS( "gui.viewer.volume_rendering.bounding_box_enabled", bool, true ) );

		Vector4d color = GET_SETTINGS( "gui.viewer.background_color", Vector4d, Vector4d( 0.0, 0.0, 0.0, 1.0 ) );
		viewer->setBackgroundColor( QColor::fromRgbF( color[0], color[1], color[2], color[3] ) );


		//viewer->setBackgroundColor( QColor( 50,50,100 ) );
		return viewer;
	}
	void
	setRenderingExtension( RenderingExtension::Ptr aRenderingExtension )
	{
		mRenderingExtension = aRenderingExtension;
	}

	void
	setViewerController( AViewerController::Ptr aController )
	{
		mViewerController = aController;
	}
	void
	setPrimaryInputConnection( M4D::Imaging::ConnectionInterface &mProdconn )
	{
		mPrimaryConnection = &mProdconn;
	}
	void
	setSecondaryInputConnection( M4D::Imaging::ConnectionInterface &mProdconn )
	{
		mSecondaryConnection = &mProdconn;
	}
protected:
	RenderingExtension::Ptr mRenderingExtension;
	AViewerController::Ptr	mViewerController;
	M4D::Imaging::ConnectionInterface *mPrimaryConnection;
	M4D::Imaging::ConnectionInterface *mSecondaryConnection;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/


bool fillBufferFromTF(M4D::GUI::TransferFunctionInterface::Const function, vorgl::TransferFunctionBuffer1D::Ptr& buffer){

	if(!function) return false;

	M4D::GUI::TF::Size domain = function.getDomain(TF_DIMENSION_1);
	if(!buffer || buffer->size() != domain)
	{
		buffer = vorgl::TransferFunctionBuffer1D::Ptr(new vorgl::TransferFunctionBuffer1D(domain, vorgl::TransferFunctionBuffer1D::MappedInterval(0.0f, (float)domain)));
	}

	M4D::GUI::TF::Coordinates coords(1);
	M4D::GUI::TF::Color color;
	for(M4D::GUI::TF::Size i = 0; i < domain; ++i)
	{
		coords[0] = i;
		color = function.getRGBfColor(coords);

		(*buffer)[i] = vorgl::TransferFunctionBuffer1D::value_type(
			color.component1,
			color.component2,
			color.component3,
			color.alpha);
	}
	return true;
}

bool fillIntegralBufferFromTF(M4D::GUI::TransferFunctionInterface::Const function, vorgl::TransferFunctionBuffer1D::Ptr& buffer){

	if(!function) return false;

	M4D::GUI::TF::Size domain = function.getDomain(TF_DIMENSION_1);
	if(!buffer || buffer->size() != domain)
	{
		buffer = vorgl::TransferFunctionBuffer1D::Ptr(new vorgl::TransferFunctionBuffer1D(domain, vorgl::TransferFunctionBuffer1D::MappedInterval(0.0f, (float)domain)));
	}

	M4D::GUI::TF::Coordinates coords(1);
	M4D::GUI::TF::Color lastColor = function.getRGBfColor(coords);
	M4D::GUI::TF::Color color;
	for(M4D::GUI::TF::Size i = 1; i < domain; ++i)
	{
		coords[0] = i;
		color = function.getRGBfColor(coords);

		M4D::GUI::TF::Color tmpColor = (lastColor + color)*0.5f;
		float alpha = 1.0f;//tmpColor.alpha;
		(*buffer)[i] = (*buffer)[i-1] + vorgl::TransferFunctionBuffer1D::value_type(
			tmpColor.component1 * alpha,
			tmpColor.component2 * alpha,
			tmpColor.component3 * alpha,
			tmpColor.alpha);;
		lastColor = color;
	}
	return true;
}

void
loadAllSavedTFEditorsIntoPalette( M4D::GUI::Palette &palette, boost::filesystem::path dirName )
{
	if (!boost::filesystem::exists(dirName)) {
		LOG( "Directory \'" << dirName << "\' doesn't exist!" );
		return;
	}
	if (!boost::filesystem::is_directory(dirName) ){
		LOG( "\'" << dirName << "\' is not a directory!" );
		return;
	}

	boost::filesystem::directory_iterator dirIt(dirName);
	boost::filesystem::directory_iterator end;
	for ( ;dirIt != end; ++dirIt ) {
		LOG( "Found TFE file :" << *dirIt );
		boost::filesystem::path p = dirIt->path();
		palette.loadFromFile( QString( p.string().data() ), false );
	}
}


ViewerWindow::ViewerWindow()
{
	setupUi( this );
	setAnimated( false );

	#ifdef WIN32
		//Reposition console window
		QRect myRegion=frameGeometry();
		QPoint putAt=myRegion.topRight();
		SetWindowPos(GetConsoleWindow(),(HWND)winId(),putAt.x()+1,putAt.y(),0,0,SWP_NOSIZE);
	#endif


	boost::filesystem::path dataDirName = GET_SETTINGS( "application.data_directory", std::string, (boost::filesystem::current_path() / "data").string() );
	M4D::GUI::Renderer::gSliceRendererShaderPath = dataDirName / "shaders";
	M4D::GUI::Renderer::gVolumeRendererShaderPath = dataDirName / "shaders";
	//M4D::gPickingShaderPath = dataDirName / "shaders" / "PickingShader.cgfx";

	mViewerController = ProxyViewerController::Ptr( new ProxyViewerController );
	mRenderingExtension = ProxyRenderingExtension::Ptr( new ProxyRenderingExtension );
//***********************************************************
	std::vector<M4D::GUI::TF::Size> dataCT1D(1, 4096);	//default CT
	mTFEditingSystem = M4D::GUI::Palette::Ptr(new M4D::GUI::Palette(this, dataCT1D));
	mTFEditingSystem->setupDefault();

	createDockWidget( tr("Transfer Function Palette"), Qt::RightDockWidgetArea, mTFEditingSystem.get(), true );
	/*QDockWidget* dockWidget = new QDockWidget("Transfer Function Palette", this);

	dockWidget->setWidget( &(*mTFEditingSystem) );
	dockWidget->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);

	addDockWidget(Qt::LeftDockWidgetArea, dockWidget);*/
	//dockWidget->setFloating(true);

	loadAllSavedTFEditorsIntoPalette( *mTFEditingSystem, GET_SETTINGS( "gui.transfer_functions.load_path", std::string, std::string( "./data/TF" ) ) );

	LOG( "TF framework initialized" );
//*****************

	/*mTransferFunctionEditor = new M4D::GUI::TransferFunction1DEditor;
	createDockWidget( tr("Transfer Function" ), Qt::RightDockWidgetArea, mTransferFunctionEditor );

	mTransferFunctionEditor->SetValueInterval( 0.0f, 3000.0f );
	mTransferFunctionEditor->SetMappedValueInterval( 0.0f, 1.0f );
	mTransferFunctionEditor->SetBorderWidth( 10 );

	mTransFuncTimer.setInterval( 500 );
	QObject::connect( &mTransFuncTimer, SIGNAL( timeout() ), this, SLOT( updateTransferFunction() ) );*/
//***********************************************************
#ifdef USE_PYTHON
	M4D::GUI::TerminalWidget *mTerminal = new M4D::GUI::PythonTerminal;
	createDockWidget( tr("Python Terminal" ), Qt::BottomDockWidgetArea, mTerminal, false );
	LOG( "Python terminal initialized" );
#endif //USE_PYTHON

	ViewerControls *mViewerControls = new ViewerControls;
	QObject::connect(
			ApplicationManager::getInstance(),
			&ApplicationManager::viewerSelectionChanged,
			[mViewerControls] () {
				auto viewer = ViewerManager::getInstance()->getSelectedViewer();
				auto genViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (viewer);
				mViewerControls->setViewer(genViewer);
			});
	QObject::connect( ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), mViewerControls, SLOT( updateControls() ) );
	createDockWidget( tr("Viewer Controls" ), Qt::RightDockWidgetArea, mViewerControls );
	LOG( "Viewer controls GUI initialized" );

	//************* TOOLBAR & MENU *****************
	ViewerActionSet &actions = ViewerManager::getInstance()->getViewerActionSet();
	QToolBar *toolbar = createToolBarFromViewerActionSet( actions, "Viewer settings" );
	addToolBar( toolbar );

	QAction *action = new QAction( "Denoise", this );
	QObject::connect( action, SIGNAL( triggered(bool) ), this, SLOT( denoiseImage() ) );
	toolbar = new QToolBar( "Image processing", this );
	toolbar->addAction( action );
	addToolBar( toolbar );



	addViewerActionSetToWidget( *menuViewer, actions );
	LOG( "Viewer control actions created and added to GUI" );

	mInfoLabel = new QLabel();
	statusbar->addWidget( mInfoLabel );


	//************* VIEWER FACTORY *****************
	M4D::GUI::Viewer::GeneralViewerFactory::Ptr factory = M4D::GUI::Viewer::GeneralViewerFactory::Ptr( new M4D::GUI::Viewer::GeneralViewerFactory );
	factory->setViewerController( mViewerController );
	factory->setRenderingExtension( mRenderingExtension );
	factory->setPrimaryInputConnection( DatasetManager::getInstance()->primaryImageInputConnection() );
	factory->setSecondaryInputConnection( DatasetManager::getInstance()->secondaryImageInputConnection() );
	mViewerDesktop->setViewerFactory( factory );
	LOG( "Viewer factory initialized" );

	mViewerDesktop->setLayoutOrganization( 2, 1 );
	QObject::connect( mViewerDesktop, SIGNAL( updateInfo( const QString & ) ), this, SLOT( updateInfoInStatusBar( const QString & ) ), Qt::QueuedConnection );

	QObject::connect( this, SIGNAL( callInitAfterLoopStart() ), this, SLOT( initAfterLoopStart() ), Qt::QueuedConnection );
	emit callInitAfterLoopStart();



	// HH: OIS support
#ifdef OIS_ENABLED
	mJoyInput.startup((size_t)this->winId());
	if(mJoyInput.getNrJoysticks() > 0) {
		mJoyTimer.setInterval( 50 );
		QObject::connect( &mJoyTimer, SIGNAL( timeout() ), this, SLOT( updateJoyControl() ) );
		mJoyTimer.start();
	}
	LOG( "OIS initialized" );
#endif


	QObject::connect( ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), this, SLOT( selectedViewerSettingsChanged() ) );

	QObject::connect( ApplicationManager::getInstance(), SIGNAL( viewerSelectionChanged() ), this, SLOT( changedViewerSelection() ) );

	QObject::connect( mTFEditingSystem.get(), SIGNAL(transferFunctionAdded( int ) ), this, SLOT( transferFunctionAdded( int ) ), Qt::QueuedConnection );
	QObject::connect( mTFEditingSystem.get(), SIGNAL(changedTransferFunctionSelection( int ) ), this, SLOT( changedTransferFunctionSelection() ), Qt::QueuedConnection );
	QObject::connect( mTFEditingSystem.get(), SIGNAL(transferFunctionModified( int )), this, SLOT( transferFunctionModified( int ) ), Qt::QueuedConnection );

	LOG( "Basic signals/slots connected" );

	mOpenDialog = new QFileDialog( this, tr("Open Image") );
	mOpenDialog->setFileMode(QFileDialog::ExistingFile);
	//mOpenDialog->setOption(QFileDialog::DontUseNativeDialog, false);
}

void
ViewerWindow::denoiseImage()
{
	LOG( "denoiseImage()" );
	ImageRecord::Ptr rec = DatasetManager::getInstance()->getCurrentImageInfo();
	M4D::Imaging::AImage::Ptr image = rec->image;
	if( !image ) {
		return;
	}
	#ifdef USE_CUDA
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			IMAGE_TYPE::Ptr outputImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< TTYPE, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			median3D( typedImage->GetRegion(), outputImage->GetRegion(), 2 );
			DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( outputImage );
			rec->image = outputImage;
		);
	#endif
		//TODO - CPU version
}

void
ViewerWindow::initAfterLoopStart()
{
	//toggleInteractiveTransferFunction( true );
}


ViewerWindow::~ViewerWindow()
{
#ifdef OIS_ENABLED
	mJoyInput.destroy();
#endif

}

void
ViewerWindow::updateInfoInStatusBar( const QString &aInfo )
{
	statusbar->showMessage( aInfo ) ;
}

M4D::GUI::Viewer::GeneralViewer *
ViewerWindow::getSelectedViewer()
{
	return NULL;
}

void
ViewerWindow::addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension )
{
	mRenderingExtension->addRenderingExtension( aRenderingExtension );
}

void
ViewerWindow::setViewerController( M4D::GUI::Viewer::AViewerController::Ptr aViewerController )
{
	mViewerController->setController( aViewerController );
}

void
ViewerWindow::testSlot()
{
	LOG( "AAAAA" );
	updateGui();

	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	if(pViewer != NULL) {
		//pViewer->toggleFPS();
		static_cast<M4D::GUI::Viewer::GeneralViewer *>(pViewer)->reloadShaders();
	}
}

void
ViewerWindow::updateGui()
{
	mViewerDesktop->updateAllViewers();
}


struct SetTransferFunctionFtor
{
	SetTransferFunctionFtor( vorgl::TransferFunctionBuffer1D::Ptr aTF ): mTF( aTF )
	{ /*empty*/ }

	void
	operator()( M4D::GUI::Viewer::AGLViewer * aViewer )
	{
		M4D::GUI::Viewer::GeneralViewer * viewer = dynamic_cast< M4D::GUI::Viewer::GeneralViewer * >( aViewer );

		if ( viewer ) {
			viewer->setTransferFunctionBuffer( mTF );
		}
	}

	vorgl::TransferFunctionBuffer1D::Ptr mTF;
};

void
ViewerWindow::applyTransferFunction()
{
	mViewerDesktop->forEachViewer( SetTransferFunctionFtor( mTransferFunctionEditor->GetTransferFunctionBuffer() ) );
}

void
ViewerWindow::updateTransferFunction()
{
	M4D::Common::TimeStamp timestamp = mTransferFunctionEditor->GetTimeStamp();
	if ( timestamp != mLastTimeStamp ) {
		applyTransferFunction();
		mLastTimeStamp = timestamp;
	}
}

#ifdef OIS_ENABLED
void
ViewerWindow::updateJoyControl()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	static M4D::Common::Clock myTimer;

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	if(pGenViewer != NULL) {
		mJoyInput.updateJoys();
		float dt = myTimer.SecondsPassed();
		myTimer.Reset();

		float fConstant = dt * 4.0f;
		static float alpha = 0, beta = 0;

		int iOffset = 512;
		int iInputValue = mJoyInput.getSlider(0, 0);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, int(0)): M4D::max(iInputValue - iOffset, int(0));
		beta = (beta) * (1-fConstant) + fConstant * (-iInputValue * M_PI / 32768);
		iInputValue = mJoyInput.getAxis(0, 0);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, 0): M4D::max(iInputValue - iOffset, 0);
		alpha = (alpha) * (1-fConstant) + fConstant * (iInputValue * M_PI / 32768);
		pGenViewer->cameraOrbit(Vector2f(alpha*0.05f, beta*0.05f));

		float camZ1 = 0;
		iInputValue = mJoyInput.getAxis(0, 1);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, 0): M4D::max(iInputValue - iOffset, 0); // right stick Y
		camZ1 = fConstant * iInputValue / 65535.0f;
		float camZ2 = 0;
		iInputValue = mJoyInput.getAxis(0, 4);
		iInputValue = (iInputValue < 0)? M4D::min(iInputValue + iOffset, 0): M4D::max(iInputValue - iOffset, 0); // left stick Y
		camZ2 = fConstant * iInputValue / 65535.0f;
		pGenViewer->cameraDolly(1.0f + camZ1 + camZ2);
	}
}
#endif

void
ViewerWindow::toggleInteractiveTransferFunction( bool aChecked )
{
	if ( aChecked ) {
		D_PRINT( "Transfer function - interactive manipulation enabled" );
		mTransFuncTimer.start();
	} else {
		D_PRINT( "Transfer function - interactive manipulation disabled" );
		mTransFuncTimer.stop();
	}
}


void
ViewerWindow::selectedViewerSettingsChanged()
{
	/*M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	QString text;
	if(pGenViewer != NULL) {
		if( pGenViewer->getViewType() == M4D::GUI::Viewer::vt2DAlignedSlices ) {
			if( pGenViewer->getColorTransformType() == M4D::GUI::Renderer::ctLUTWindow ) {
				Vector2f win = pGenViewer->getLUTWindow();
				text = QString::number( win[0] ) + "/" + QString::number( win[1] );
			}
		}
	}
	mInfoLabel->setText( text );*/
	//LOG( __FUNCTION__ );
}


void
ViewerWindow::changedViewerSelection()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	if(pGenViewer != NULL) {
		vorgl::TransferFunctionBufferInfo info = pGenViewer->getTransferFunctionBufferInfo();
		mTFEditingSystem->selectTransferFunction( info.id );
	}
	//LOG( __FUNCTION__ );
}

/*struct CreateGLTFBuffer
{
	vorgl::GLTransferFunctionBuffer1D::Ptr tfGLBuffer;
	vorgl::TransferFunctionBuffer1D::Ptr tfBuffer;

	void
	operator()()
	{
		 tfGLBuffer = createGLTransferFunctionBuffer1D( *tfBuffer );
	}
};*/

bool
fillTransferFunctionInfo( M4D::GUI::TransferFunctionInterface::Const function, vorgl::TransferFunctionBufferInfo &info )
{
	if( fillBufferFromTF( function, info.tfBuffer ) ) {
		//std::ofstream file( "TF.txt" );
		//D_PRINT( "Printing TF" );
		//std::copy( info.tfBuffer->Begin(), info.tfBuffer->End(), std::ostream_iterator<M4D::GUI::TransferFunctionBuffer1D::value_type>( file, " | " ) );

		//CreateGLTFBuffer ftor;
		//ftor.tfBuffer = info.tfBuffer;
		//ftor = OpenGLManager::getInstance()->doGL( ftor );
		//info.tfGLBuffer = ftor.tfGLBuffer;

		OpenGLManager::getInstance()->doGL([&info]() {
				info.tfGLBuffer = createGLTransferFunctionBuffer1D( *info.tfBuffer );
			});

		if( fillIntegralBufferFromTF( function, info.tfIntegralBuffer ) )
		{
			//file << std::endl;
			//std::copy( info.tfIntegralBuffer->Begin(), info.tfIntegralBuffer->End(), std::ostream_iterator<M4D::GUI::TransferFunctionBuffer1D::value_type>( file, " | " ) );

			/*ftor.tfBuffer = info.tfIntegralBuffer;
			ftor = OpenGLManager::getInstance()->doGL( ftor );
			info.tfGLIntegralBuffer = ftor.tfGLBuffer;*/
			OpenGLManager::getInstance()->doGL([&info]() {
				info.tfGLIntegralBuffer = createGLTransferFunctionBuffer1D( *info.tfIntegralBuffer );
			});
		} else {
			OpenGLManager::getInstance()->doGL([&info]() {
				info.tfGLIntegralBuffer = vorgl::GLTransferFunctionBuffer1D::Ptr();
			});
		}
		//file.close();
	} else {
		D_PRINT( "TF buffer not created" );
		return false;
	}
	return true;
}

void
ViewerWindow::transferFunctionAdded( int idx )
{
	vorgl::TransferFunctionBufferInfo info;
	info.id = idx;

	if ( fillTransferFunctionInfo( mTFEditingSystem->getTransferFunction(idx), info ) ) {
		TransferFunctionBufferUsageRecord record;
		record.info = info;
		mTFUsageMap[idx] = record;
	}
}

void
ViewerWindow::changedTransferFunctionSelection()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
	pViewer = ViewerManager::getInstance()->getSelectedViewer();

	M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
	if(pGenViewer != NULL) {
		M4D::Common::IDNumber idx = mTFEditingSystem->getActiveEditorId();
		vorgl::TransferFunctionBufferInfo oldInfo = pGenViewer->getTransferFunctionBufferInfo();

		if ( idx == oldInfo.id ) {
			return; //No change
		}
		TransferBufferUsageMap::iterator it = mTFUsageMap.find( oldInfo.id );
		if ( it != mTFUsageMap.end() ) {
			it->second.viewers.remove( pGenViewer );
		}
		it = mTFUsageMap.find( idx );
		if( it == mTFUsageMap.end() ) {
			transferFunctionAdded( idx );
		}
		it = mTFUsageMap.find( idx );
		if ( it != mTFUsageMap.end() ) {
			pGenViewer->setTransferFunctionBufferInfo( it->second.info );
			it->second.viewers.push_back( pGenViewer );
		} else {
			LOG( "Function not found" );
		}
	}
	//LOG( __FUNCTION__ );
}

void
ViewerWindow::transferFunctionModified( int idx )
{
	TransferBufferUsageMap::iterator it = mTFUsageMap.find( idx );
	if ( it != mTFUsageMap.end() ) {
		TransferFunctionBufferUsageRecord &rec = it->second;

		fillTransferFunctionInfo( mTFEditingSystem->getTransferFunction(idx), rec.info );
		for (ViewerList::iterator it = rec.viewers.begin(); it != rec.viewers.end(); ++it ) {
			(*it)->setTransferFunctionBufferInfo( rec.info );
		}
	} else {
		D_PRINT( "Modified function not found" );
	}
	//LOG( __FUNCTION__ );
}


void
ViewerWindow::showSettingsDialog()
{
	SettingsDialog *dialog = new SettingsDialog( this );
	dialog->showDialog( ApplicationManager::getInstance()->settings() );
	delete dialog;
}

void
ViewerWindow::openFile()
{
	try {
	QStringList fileNames;
	QString fileName;
	if ( mOpenDialog->exec() ) {
		fileNames = mOpenDialog->selectedFiles();
		if (!fileNames.isEmpty()) {
			fileName = fileNames[0];
		}
	}
	//QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image") );

	/*if ( !fileName.isEmpty() ) {
		QFileInfo pathInfo( fileName );
		QString ext = pathInfo.suffix();
		if ( ext.toLower() == "dcm" ) {
			openDicom( fileName );
		} else {
			openFile( fileName );
		}
	}*/

	if ( !fileName.isEmpty() ) {
		if( !mProgressDialog ) {
			mProgressDialog = ProgressInfoDialog::Ptr( new ProgressInfoDialog( this ) );
			QObject::connect( mProgressDialog.get(), SIGNAL( finishedSignal() ), this, SLOT( dataLoaded() ), Qt::QueuedConnection );
		}

		mDatasetId = DatasetManager::getInstance()->openFileNonBlocking( std::string( fileName.toLocal8Bit().data() ), mProgressDialog );

		mProgressDialog->show();
	}
	} catch ( std::exception &e ) {
		QMessageBox::critical ( NULL, "Exception", QString( e.what() ) );
	}
	catch (...) {
		QMessageBox::critical ( NULL, "Exception", "Problem with file loading" );
	}
}

void
ViewerWindow::openFile( const QString &aPath )
{
	ASSERT( false );
	std::string path = std::string( aPath.toLocal8Bit().data() );
	M4D::Imaging::AImage::Ptr image = M4D::Imaging::ImageFactory::LoadDumpedImage( path );
	DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( image );

	//M4D::Common::Clock clock;

	//M4D::Imaging::Histogram64::Ptr histogram = M4D::Imaging::Histogram64::Create( 0, 4065, true );
	/*IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		M4D::Imaging::AddRegionToHistogram( *histogram, IMAGE_TYPE::Cast( image )->GetRegion() );
	);*/
	/*IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		IMAGE_TYPE::PointType strides;
		IMAGE_TYPE::SizeType size;
		IMAGE_TYPE::Element *pointer = IMAGE_TYPE::Cast( image )->GetPointer( size, strides );
		M4D::Imaging::AddArrayToHistogram( *histogram, pointer, VectorCoordinateProduct( size )  );
	);*/

/*	M4D::Imaging::Histogram64::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		histogram = M4D::Imaging::CreateHistogramForImageRegion<M4D::Imaging::Histogram64, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);

	LOG( "Histogram computed in " << clock.SecondsPassed() );
	mTransferFunctionEditor->SetBackgroundHistogram( histogram );
*/

	//applyTransferFunction();
	//
	typedef M4D::Imaging::SimpleHistogram64 Histogram;
	typedef M4D::GUI::TF::Histogram<1> TFHistogram;

	statusbar->showMessage("Computing histogram...");
	M4D::Common::Clock clock;

	Histogram::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		histogram = M4D::Imaging::createHistogramForImageRegion<Histogram, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);

	M4D::Imaging::Histogram1D<int> histogram2;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		histogram2 = M4D::Imaging::createHistogramForImageRegion2<M4D::Imaging::Histogram1D<int>, IMAGE_TYPE>(IMAGE_TYPE::Cast(*image));
	);

	int domain = mTFEditingSystem->getDomain(TF_DIMENSION_1);

	TFHistogram* tfHistogram(new TFHistogram(std::vector<M4D::GUI::TF::Size>(1, domain)));

	M4D::GUI::TF::Coordinates coords(1, histogram->getMin());
	for(Histogram::iterator it = histogram->begin(); it != histogram->end(); ++it)	//values
	{
		tfHistogram->set(coords, *it);
		++coords[0];
	}
	tfHistogram->seal();

	mTFEditingSystem->setHistogram(M4D::GUI::TF::HistogramInterface::Ptr(tfHistogram));

	LOG( "Histogram computed in " << clock.SecondsPassed() );

	//statusbar->showMessage("Applying transfer function...");
	//applyTransferFunction();

	statusbar->clearMessage();
}

void
ViewerWindow::openDicom( const QString &aPath )
{
	ASSERT( false );
	LOG( "Opening Dicom" );
	std::string path = std::string( aPath.toLocal8Bit().data() );

	QFileInfo pathInfo( aPath );

	if( !mDicomObjSet ) {
		mDicomObjSet = M4D::Dicom::DicomObjSetPtr( new M4D::Dicom::DicomObjSet() );
	}

	if( !mProgressDialog ) {
		mProgressDialog = ProgressInfoDialog::Ptr( new ProgressInfoDialog( this ) );
		QObject::connect( mProgressDialog.get(), SIGNAL( finishedSignal() ), this, SLOT( dataLoaded() ), Qt::QueuedConnection );
	}
	/*boost::thread th = boost::thread(
			&M4D::Dicom::DcmProvider::LoadSerieThatFileBelongsTo,
			std::string( pathInfo.absoluteFilePath().toLocal8Bit().data() ),
			std::string( pathInfo.absolutePath().toLocal8Bit().data() ),
			boost::ref( *mDicomObjSet ),
			mProgressDialog
			);
	th.detach();*/
	mProgressDialog->show();
	//th.join();
}

void
ViewerWindow::dataLoaded()
{
	//M4D::Imaging::AImage::Ptr image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( mDicomObjSet );
	D_PRINT( "Loaded dataset ID = " << mDatasetId );
	ADatasetRecord::Ptr rec = DatasetManager::getInstance()->getDatasetInfo( mDatasetId );
	if ( !rec ) {
		D_PRINT( "Loaded dataset record not available" );
		return;
	}
	ImageRecord * iRec = dynamic_cast< ImageRecord * >( rec.get() );
	if ( !iRec ) {
		D_PRINT( "Loaded dataset isn't image" );
	}
	M4D::Imaging::AImage::Ptr image = iRec->image;

	DatasetManager::getInstance()->primaryImageInputConnection().PutDataset( image );

	//M4D::Common::Clock clock;

	//M4D::Imaging::Histogram64::Ptr histogram = M4D::Imaging::Histogram64::Create( 0, 4065, true );
	/*IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		M4D::Imaging::AddRegionToHistogram( *histogram, IMAGE_TYPE::Cast( image )->GetRegion() );
	);*/
	/*IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		IMAGE_TYPE::PointType strides;
		IMAGE_TYPE::SizeType size;
		IMAGE_TYPE::Element *pointer = IMAGE_TYPE::Cast( image )->GetPointer( size, strides );
		M4D::Imaging::AddArrayToHistogram( *histogram, pointer, VectorCoordinateProduct( size )  );
	);*/

	/*M4D::Imaging::Histogram64::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image,
		histogram = M4D::Imaging::CreateHistogramForImageRegion<M4D::Imaging::Histogram64, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);

	LOG( "Histogram computed in " << clock.SecondsPassed() );
	mTransferFunctionEditor->SetBackgroundHistogram( histogram );*/


	//applyTransferFunction();

	//boost::thread th = boost::thread( boost::bind( &ViewerWindow::computeHistogram, this, image ) );
	//th.detach();

	computeHistogram( image );

}

void
ViewerWindow::computeHistogram( M4D::Imaging::AImage::Ptr aImage )
{
	if( !aImage ) {
		return;
	}
	typedef M4D::Imaging::SimpleHistogram64 Histogram;
	typedef M4D::GUI::TF::Histogram<1> TFHistogram;

	statusbar->showMessage("Computing histogram...");
	M4D::Common::Clock clock;

	Histogram::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( aImage,
		histogram = M4D::Imaging::createHistogramForImageRegion<Histogram, IMAGE_TYPE >( IMAGE_TYPE::Cast( *aImage ) );
	);

	int domain = mTFEditingSystem->getDomain(TF_DIMENSION_1);

	TFHistogram* tfHistogram(new TFHistogram(std::vector<M4D::GUI::TF::Size>(1, domain)));

	M4D::GUI::TF::Coordinates coords(1, histogram->getMin());
	for(Histogram::iterator it = histogram->begin(); it != histogram->end(); ++it)	//values
	{
		tfHistogram->set(coords, *it);
		++coords[0];
	}
	tfHistogram->seal();

	mTFEditingSystem->setHistogram(M4D::GUI::TF::HistogramInterface::Ptr(tfHistogram));



	LOG( "Histogram computed in " << clock.SecondsPassed() );


	//statusbar->showMessage("Applying transfer function...");
	//applyTransferFunction();

	statusbar->clearMessage();
}
