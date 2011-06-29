#include "ViewerWindow.hpp"
//#include "GUI/utils/ImageDataRenderer.h"
#include "Imaging/ImageTools.h"
#include "Imaging/Histogram.h"
#include <cmath>

#include "GUI/widgets/PythonTerminal.h"
#include "GUI/widgets/MultiDockWidget.h"
#include "GUI/utils/ViewerManager.h"
#include "GUI/utils/ViewerAction.h"

#include <boost/thread.hpp>


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
	GeneralViewerFactory() :mConnection( NULL )
	{}
	typedef boost::shared_ptr< GeneralViewerFactory > Ptr;

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
		if( mConnection ) {
			mConnection->ConnectConsumer( viewer->InputPort()[0] );
		}

		viewer->setLUTWindow( Vector2f( 500.0f,1000.0f ) );

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
	setInputConnection( M4D::Imaging::ConnectionInterface &mProdconn )
	{
		mConnection = &mProdconn;
	}

protected:
	RenderingExtension::Ptr mRenderingExtension;
	AViewerController::Ptr	mViewerController;
	M4D::Imaging::ConnectionInterface *mConnection;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

ViewerWindow::ViewerWindow()
{
	setupUi( this );
	setAnimated( false );

	#ifdef WIN32
		//Reposition console window
		QRect myRegion=frameGeometry();
		QPoint putAt=myRegion.topRight();
		SetWindowPos(GetConsoleWindow(),winId(),putAt.x()+1,putAt.y(),0,0,SWP_NOSIZE);
	#endif

	
	MultiDockWidget * dockwidget = new MultiDockWidget( tr("Transfer Function" ) );
	mTransferFunctionEditor = new M4D::GUI::TransferFunction1DEditor;
	dockwidget->setWidget( mTransferFunctionEditor );
	dockwidget->addDockingWindow( Qt::RightDockWidgetArea, this );

	/*mMainWin2 = new QMainWindow();
	mMainWin2->show();
	dockwidget->addDockingWindow( mMainWin2 );*/


	mTransferFunctionEditor->SetValueInterval( 0.0f, 3000.0f );
	mTransferFunctionEditor->SetMappedValueInterval( 0.0f, 1.0f );
	mTransferFunctionEditor->SetBorderWidth( 10 );
	//addDockWidget (Qt::RightDockWidgetArea, dockwidget );

#ifdef USE_PYTHON
	dockwidget = new MultiDockWidget(tr("Python Terminal" ));
	M4D::GUI::TerminalWidget *mTerminal = new M4D::GUI::PythonTerminal;
	dockwidget->setWidget( mTerminal );
	dockwidget->addDockingWindow( Qt::BottomDockWidgetArea, this );
	//addDockWidget (Qt::BottomDockWidgetArea, dockwidget );
#endif //USE_PYTHON


	mTransFuncTimer.setInterval( 500 );
	QObject::connect( &mTransFuncTimer, SIGNAL( timeout() ), this, SLOT( updateTransferFunction() ) );

	/*QActionGroup *viewerTypeSwitch = new QActionGroup( this );
	QSignalMapper *viewerTypeSwitchSignalMapper = new QSignalMapper( this );
	viewerTypeSwitch->setExclusive( true );
	viewerTypeSwitch->addAction( action2D );
	viewerTypeSwitch->addAction( action3D );
	viewerTypeSwitchSignalMapper->setMapping( action2D, M4D::GUI::Viewer::vt2DAlignedSlices );
	viewerTypeSwitchSignalMapper->setMapping( action3D, M4D::GUI::Viewer::vt3D );
	QObject::connect( action2D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( action3D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( viewerTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeViewerType( int ) ) );
	*/

	mViewerController = AnnotationEditorController::Ptr( new AnnotationEditorController );

	//************* TOOLBAR & MENU *****************
	ViewerActionSet &actions = ViewerManager::getInstance()->getViewerActionSet();
	QToolBar *toolbar = createToolBarFromViewerActionSet( actions, "Viewer settings" );
	addToolBar( toolbar );

	addViewerActionSetToWidget( *menuViewer, actions );
	//toolbar->addAction( "BLLLL" );

	//**********************************************
	//************* ANNOTATION TOOLBAR *************	
	QList<QAction*> annotationActions = mViewerController->getActions();
	toolbar = new QToolBar( tr( "Annotations" ), this );
	for ( QList<QAction*>::iterator it = annotationActions.begin(); it != annotationActions.end(); ++it )
	{
		toolbar->addAction( *it );
	}
	addToolBar( toolbar );

	//addViewerActionSetToWidget( *menuViewer, actions );
	//toolbar->addAction( "BLLLL" );

	//*****************************************

	mInfoLabel = new QLabel();
	statusbar->addWidget( mInfoLabel );


	/*mColorTransformChooser = new QComboBox;
	mColorTransformChooser->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	viewerToolBar->addWidget( mColorTransformChooser );
	QObject::connect( mColorTransformChooser, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( changeColorMapType( const QString & ) ) );
	//QObject::connect( mColorTransformChooser, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( updateToolbars() ) );
	*/

	M4D::GUI::Viewer::GeneralViewerFactory::Ptr factory = M4D::GUI::Viewer::GeneralViewerFactory::Ptr( new M4D::GUI::Viewer::GeneralViewerFactory );
	factory->setViewerController( mViewerController );
	factory->setRenderingExtension( mViewerController );
	factory->setInputConnection( mProdconn );

	mViewerDesktop->setViewerFactory( factory );
	mViewerDesktop->setLayoutOrganization( 2, 1 );


	QObject::connect( this, SIGNAL( callInitAfterLoopStart() ), this, SLOT( initAfterLoopStart() ), Qt::QueuedConnection );
	emit callInitAfterLoopStart();

	QObject::connect( ApplicationManager::getInstance(), SIGNAL( selectedViewerSettingsChanged() ), this, SLOT( selectedViewerSettingsChanged() ) );

	// HH: OIS support
#ifdef OIS_ENABLED
	mJoyInput.startup((size_t)this->winId());
	if(mJoyInput.getNrJoysticks() > 0) {
		mJoyTimer.setInterval( 50 );
		QObject::connect( &mJoyTimer, SIGNAL( timeout() ), this, SLOT( updateJoyControl() ) );
		mJoyTimer.start();
	}
#endif
}

void
ViewerWindow::initAfterLoopStart()
{
	changeViewerType( M4D::GUI::Viewer::vt2DAlignedSlices );
	updateToolbars();

	toggleInteractiveTransferFunction( true );
}


ViewerWindow::~ViewerWindow()
{
#ifdef OIS_ENABLED
	mJoyInput.destroy();
#endif

}

M4D::GUI::Viewer::GeneralViewer *
ViewerWindow::getSelectedViewer()
{
	return NULL;
}


void
ViewerWindow::changeViewerType( int aViewType )
{
	M4D::GUI::Viewer::GeneralViewer * viewer = getSelectedViewer();
	if ( viewer ) { viewer->setViewType( aViewType ); }

	updateToolbars();
}

void
ViewerWindow::changeColorMapType( const QString & aColorMapName )
{
	static bool locked = false;
	if ( locked ) return;
	locked = true;

	M4D::GUI::Viewer::GeneralViewer * viewer = getSelectedViewer();
	if ( viewer ) { viewer->setColorTransformType( aColorMapName ); }
	updateToolbars();

	locked = false;
}

void
ViewerWindow::testSlot()
{
	mViewerController->mOverlay = !mViewerController->mOverlay;
	M4D::GUI::Viewer::GeneralViewer * viewer = getSelectedViewer();
	if ( viewer ) { viewer->update(); }
	
	
	
	//QImage image = mViewer->RenderThumbnailImage( QSize( 256, 256 ) );
	//label->setPixmap( QPixmap::fromImage( image ) );
}


struct SetTransferFunctionFtor
{
	SetTransferFunctionFtor( M4D::GUI::TransferFunctionBuffer1D::Ptr aTF ): mTF( aTF )
	{ /*empty*/ }

	void
	operator()( M4D::GUI::Viewer::AGLViewer * aViewer )
	{
		M4D::GUI::Viewer::GeneralViewer * viewer = dynamic_cast< M4D::GUI::Viewer::GeneralViewer * >( aViewer );

		if ( viewer ) {
			viewer->setTransferFunctionBuffer( mTF );
		}
	}

	M4D::GUI::TransferFunctionBuffer1D::Ptr mTF;
};

void
ViewerWindow::applyTransferFunction()
{
	//D_PRINT( "Function updated" );
	mViewerDesktop->forEachViewer( SetTransferFunctionFtor( mTransferFunctionEditor->GetTransferFunctionBuffer() ) );
	/*M4D::GUI::Viewer::GeneralViewer * viewer = getSelectedViewer();
	if ( viewer ) { viewer->setTransferFunctionBuffer( mTransferFunctionEditor->GetTransferFunctionBuffer() ); }*/
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
		iInputValue = (iInputValue < 0)?Min(iInputValue + iOffset, 0):Max(iInputValue - iOffset, 0);
		beta = (beta) * (1-fConstant) + fConstant * (-iInputValue * M_PI / 32768);
		iInputValue = mJoyInput.getAxis(0, 0);
		iInputValue = (iInputValue < 0)?Min(iInputValue + iOffset, 0):Max(iInputValue - iOffset, 0);
		alpha = (alpha) * (1-fConstant) + fConstant * (iInputValue * M_PI / 32768);
		pGenViewer->cameraOrbit(Vector2f(alpha*0.05f, beta*0.05f));

		float camZ1 = 0;
		iInputValue = mJoyInput.getAxis(0, 1);
		iInputValue = (iInputValue < 0)?Min(iInputValue + iOffset, 0):Max(iInputValue - iOffset, 0); // right stick Y
		camZ1 = fConstant * iInputValue / 65535.0f;
		float camZ2 = 0;
		iInputValue = mJoyInput.getAxis(0, 4);
		iInputValue = (iInputValue < 0)?Min(iInputValue + iOffset, 0):Max(iInputValue - iOffset, 0); // left stick Y
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
ViewerWindow::updateToolbars()
{
	static bool locked = false;
	if ( locked ) return;
	locked = true;
	

	locked = false;
}

void
ViewerWindow::selectedViewerSettingsChanged()
{
	M4D::GUI::Viewer::AGLViewer *pViewer;
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
	mInfoLabel->setText( text );
}

void
ViewerWindow::openFile()
{
	try {
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image") );

	if ( !fileName.isEmpty() ) {
		QFileInfo pathInfo( fileName );
		QString ext = pathInfo.suffix();
		if ( ext.toLower() == "dcm" ) {
			openDicom( fileName );
		} else {
			openFile( fileName );
		}
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
	std::string path = std::string( aPath.toLocal8Bit().data() );
	M4D::Imaging::AImage::Ptr image = M4D::Imaging::ImageFactory::LoadDumpedImage( path );
	mProdconn.PutDataset( image );
	
	M4D::Common::Clock clock;
	
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

	M4D::Imaging::Histogram64::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		histogram = M4D::Imaging::CreateHistogramForImageRegion<M4D::Imaging::Histogram64, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);

	LOG( "Histogram computed in " << clock.SecondsPassed() );
	mTransferFunctionEditor->SetBackgroundHistogram( histogram );
	

	applyTransferFunction();
}

void 
ViewerWindow::openDicom( const QString &aPath )
{
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
	boost::thread th = boost::thread( 
			&M4D::Dicom::DcmProvider::LoadSerieThatFileBelongsTo,  
			std::string( pathInfo.absoluteFilePath().toLocal8Bit().data() ), 
			std::string( pathInfo.absolutePath().toLocal8Bit().data() ), 
			boost::ref( *mDicomObjSet ),
			mProgressDialog
			);
	th.detach();
	mProgressDialog->show();
	//th.join();
}

void
ViewerWindow::dataLoaded()
{
	M4D::Imaging::AImage::Ptr image = M4D::Dicom::DcmProvider::CreateImageFromDICOM( mDicomObjSet );

	mProdconn.PutDataset( image );
	
	M4D::Common::Clock clock;
	
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

	M4D::Imaging::Histogram64::Ptr histogram;
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		histogram = M4D::Imaging::CreateHistogramForImageRegion<M4D::Imaging::Histogram64, IMAGE_TYPE >( IMAGE_TYPE::Cast( *image ) );
	);

	LOG( "Histogram computed in " << clock.SecondsPassed() );
	mTransferFunctionEditor->SetBackgroundHistogram( histogram );
	

	applyTransferFunction();
}
