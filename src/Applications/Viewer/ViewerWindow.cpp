#include "ViewerWindow.hpp"
//#include "GUI/utils/ImageDataRenderer.h"
#include "Imaging/ImageTools.h"
#include "Imaging/Histogram.h"
#include <cmath>

#include "GUI/widgets/PythonTerminal.h"
#include "GUI/widgets/MultiDockWidget.h"

#include <QtGui>

ViewerWindow::ViewerWindow()
{
	setupUi( this );

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

	QActionGroup *viewerTypeSwitch = new QActionGroup( this );
	QSignalMapper *viewerTypeSwitchSignalMapper = new QSignalMapper( this );
	viewerTypeSwitch->setExclusive( true );
	viewerTypeSwitch->addAction( action2D );
	viewerTypeSwitch->addAction( action3D );
	viewerTypeSwitchSignalMapper->setMapping( action2D, M4D::GUI::Viewer::vt2DAlignedSlices );
	viewerTypeSwitchSignalMapper->setMapping( action3D, M4D::GUI::Viewer::vt3D );
	QObject::connect( action2D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( action3D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( viewerTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeViewerType( int ) ) );


	//*************** TOOLBAR ***************	
	QToolBar *toolbar = new QToolBar( tr("Viewer settings") );
	addToolBar( toolbar );
	toolbar->addAction( "BLLLL" );

	//***************************************

	QLabel *infoLabel = new QLabel();
	statusbar->addWidget( infoLabel );


	mColorTransformChooser = new QComboBox;
	mColorTransformChooser->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	viewerToolBar->addWidget( mColorTransformChooser );
	QObject::connect( mColorTransformChooser, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( changeColorMapType( const QString & ) ) );
	//QObject::connect( mColorTransformChooser, SIGNAL( currentIndexChanged( const QString & ) ), this, SLOT( updateToolbars() ) );

	mViewerController = EditorController::Ptr( new EditorController );

	/*
	mViewer->setViewerController( mViewerController );
	mViewer->setRenderingExtension( mViewerController );

	mProdconn.ConnectConsumer( mViewer->InputPort()[0] );
	mViewer->setLUTWindow( Vector2f( 500.0f,1000.0f ) );

	QObject::connect( mViewer, SIGNAL( MouseInfoUpdate( const QString & ) ), infoLabel, SLOT( setText( const QString & ) ) );
	*/

	mViewerDesktop->setLayoutOrganization( 2, 1 );

	QObject::connect( this, SIGNAL( callInitAfterLoopStart() ), this, SLOT( initAfterLoopStart() ), Qt::QueuedConnection );
	emit callInitAfterLoopStart();
}

void
ViewerWindow::initAfterLoopStart()
{
	changeViewerType( M4D::GUI::Viewer::vt2DAlignedSlices );
	updateToolbars();
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


void
ViewerWindow::applyTransferFunction()
{
	M4D::GUI::Viewer::GeneralViewer * viewer = getSelectedViewer();
	if ( viewer ) { viewer->setTransferFunctionBuffer( mTransferFunctionEditor->GetTransferFunctionBuffer() ); }
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
	
	M4D::GUI::Viewer::GeneralViewer * viewer = getSelectedViewer();
	if ( viewer ) 
	{ 
		int viewType = viewer->getViewType();
		switch ( viewType ) {
		case M4D::GUI::Viewer::vt2DAlignedSlices:
			action2D->setChecked( true );
			break;
		case M4D::GUI::Viewer::vt2DGeneralSlices:
			ASSERT( false );
			break;
		case M4D::GUI::Viewer::vt3D:
			action3D->setChecked( true );
			break;
		default:
			ASSERT( false );
		};
		
		//D_PRINT( "update toolbars" );
		mColorTransformChooser->clear();
		mColorTransformChooser->addItems( viewer->getAvailableColorTransformations() );
		int idx = mColorTransformChooser->findText( viewer->getColorTransformName() );
		//D_PRINT( "set selected color transform" );
		mColorTransformChooser->setCurrentIndex( idx );

		//LOG( "update toolbars - " << viewer->getColorTransformName().toStdString() );
		toggleInteractiveTransferFunction( viewer->getColorTransformType() == M4D::GUI::Renderer::ctTransferFunction1D );
	}

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
