#include "ViewerWindow.hpp"
//#include "GUI/utils/ImageDataRenderer.h"
#include "Imaging/ImageTools.h"
#include "Imaging/Histogram.h"
#include <cmath>

#include "GUI/widgets/PythonTerminal.h"
#include "GUI/widgets/MultiDockWidget.h"
#include "GUI/utils/ViewerManager.h"
#include "GUI/utils/ViewerAction.h"

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

	dockwidget = new MultiDockWidget(tr("Python Terminal" ));
	M4D::GUI::TerminalWidget *mTerminal = new M4D::GUI::PythonTerminal;
	dockwidget->setWidget( mTerminal );
	dockwidget->addDockingWindow( Qt::BottomDockWidgetArea, this );
	//addDockWidget (Qt::BottomDockWidgetArea, dockwidget );



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

	//************* TOOLBAR & MENU *************	
	ViewerActionSet &actions = ViewerManager::getInstance()->getViewerActionSet();
	QToolBar *toolbar = createToolBarFromViewerActionSet( actions, "Viewer settings" );
	addToolBar( toolbar );

	addViewerActionSetToWidget( *menuViewer, actions );
	//toolbar->addAction( "BLLLL" );

	//*****************************************

	QLabel *infoLabel = new QLabel();
	statusbar->addWidget( infoLabel );


	/*mColorTransformChooser = new QComboBox;
	mColorTransformChooser->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	viewerToolBar->addWidget( mColorTransformChooser );
	QObject::connect( mColorTransformChooser, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( changeColorMapType( const QString & ) ) );
	//QObject::connect( mColorTransformChooser, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( updateToolbars() ) );
	*/

	mViewerController = EditorController::Ptr( new EditorController );

	/*

	mProdconn.ConnectConsumer( mViewer->InputPort()[0] );
	mViewer->setLUTWindow( Vector2f( 500.0f,1000.0f ) );

	QObject::connect( mViewer, SIGNAL( MouseInfoUpdate( const QString & ) ), infoLabel, SLOT( setText( const QString & ) ) );
	*/
	M4D::GUI::Viewer::GeneralViewerFactory::Ptr factory = M4D::GUI::Viewer::GeneralViewerFactory::Ptr( new M4D::GUI::Viewer::GeneralViewerFactory );
	factory->setViewerController( mViewerController );
	factory->setRenderingExtension( mViewerController );
	factory->setInputConnection( mProdconn );

	mViewerDesktop->setViewerFactory( factory );
	mViewerDesktop->setLayoutOrganization( 2, 1 );


	QObject::connect( this, SIGNAL( callInitAfterLoopStart() ), this, SLOT( initAfterLoopStart() ), Qt::QueuedConnection );
	emit callInitAfterLoopStart();
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
	D_PRINT( "Function updated" );
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
ViewerWindow::openFile()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image") );

	if ( !fileName.isEmpty() ) {
		openFile( fileName );
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
