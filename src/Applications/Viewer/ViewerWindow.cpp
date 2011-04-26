#include "ViewerWindow.hpp"
//#include "GUI/utils/ImageDataRenderer.h"
#include "Imaging/ImageTools.h"
#include "Imaging/Histogram.h"
#include <cmath>

ViewerWindow::ViewerWindow()
{
	setupUi( this );

	#ifdef WIN32
		//Reposition console window
		QRect myRegion=frameGeometry();
		QPoint putAt=myRegion.topRight();
		SetWindowPos(GetConsoleWindow(),winId(),putAt.x()+1,putAt.y(),0,0,SWP_NOSIZE);
	#endif

	mProdconn.ConnectConsumer( mViewer->InputPort()[0] );
	
	QDockWidget * dockwidget = new QDockWidget;
	mTransferFunctionEditor = new M4D::GUI::TransferFunction1DEditor;
	dockwidget->setWidget( mTransferFunctionEditor );

	mTransferFunctionEditor->SetValueInterval( 0.0f, 3000.0f );
	mTransferFunctionEditor->SetMappedValueInterval( 0.0f, 1.0f );
	mTransferFunctionEditor->SetBorderWidth( 10 );
	addDockWidget (Qt::RightDockWidgetArea, dockwidget );

	mViewer->setLUTWindow( Vector2f( 500.0f,1000.0f ) );

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

	QActionGroup *colorMapTypeSwitch = new QActionGroup( this );
	QSignalMapper *colorMapTypeSwitchSignalMapper = new QSignalMapper( this );
	colorMapTypeSwitch->setExclusive( true );
	colorMapTypeSwitch->addAction( actionUse_WLWindow );
	colorMapTypeSwitch->addAction( actionUse_Simple_Colormap );
	colorMapTypeSwitch->addAction( actionUse_MIP );
	colorMapTypeSwitch->addAction( actionUse_Transfer_Function );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_WLWindow, M4D::GUI::Renderer::ctLUTWindow );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_Simple_Colormap, M4D::GUI::Renderer::ctSimpleColorMap );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_MIP, M4D::GUI::Renderer::ctMaxIntensityProjection );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_Transfer_Function, M4D::GUI::Renderer::ctTransferFunction1D );
	QObject::connect( actionUse_WLWindow, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_Simple_Colormap, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_MIP, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_Transfer_Function, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( colorMapTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeColorMapType( int ) ) );


/*	QMenu *menu = new QMenu;
	menu->addAction( actionEnable_Jittering );
	menu->addAction( actionEnable_Shading );
	
	QToolButton *button = new QToolButton;
	button->setText( "Settings" );
	button->setPopupMode( QToolButton::InstantPopup );
	button->setMenu( menu );
	viewerToolBar->addWidget( button );*/



	updateToolbars();

	QLabel *infoLabel = new QLabel();
	statusbar->addWidget( infoLabel );

	//mViewer->setMouseTracking ( true );
	QObject::connect( mViewer, SIGNAL( MouseInfoUpdate( const QString & ) ), infoLabel, SLOT( setText( const QString & ) ) );

	mColorTransformChooser = new QComboBox;
	mColorTransformChooser->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	viewerToolBar->addWidget( mColorTransformChooser );

	mViewerController = M4D::GUI::Viewer::ViewerController::Ptr( new M4D::GUI::Viewer::ViewerController );
	mViewer->setViewerController( mViewerController );

	QObject::connect( this, SIGNAL( callInitAfterLoopStart() ), this, SLOT( initAfterLoopStart() ), Qt::QueuedConnection );
	emit callInitAfterLoopStart();
}

void
ViewerWindow::initAfterLoopStart()
{
	changeViewerType( M4D::GUI::Viewer::vt2DAlignedSlices );
}


ViewerWindow::~ViewerWindow()
{

}

void
ViewerWindow::changeViewerType( int aViewType )
{
	D_PRINT( "Change viewer type" );
	if ( aViewType == M4D::GUI::Viewer::vt3D 
		&& ( mViewer->getColorTransformType() == M4D::GUI::Renderer::ctLUTWindow 
		|| mViewer->getColorTransformType() == M4D::GUI::Renderer::ctSimpleColorMap ) ) 
	{
		changeColorMapType( M4D::GUI::Renderer::ctTransferFunction1D );
	}

	if ( aViewType == M4D::GUI::Viewer::vt2DAlignedSlices 
		&& mViewer->getColorTransformType() == M4D::GUI::Renderer::ctMaxIntensityProjection ) 
	{
		changeColorMapType( M4D::GUI::Renderer::ctLUTWindow );
	}

	mViewer->setViewType( aViewType );


	mColorTransformChooser->clear();
	mColorTransformChooser->addItems( mViewer->getAvailableColorTransformations() );
	updateToolbars();
}

void
ViewerWindow::changeColorMapType( int aColorMap )
{
	D_PRINT( "Change color map type" );
	mViewer->setColorTransformType( aColorMap );
	toggleInteractiveTransferFunction( aColorMap == M4D::GUI::Renderer::ctTransferFunction1D );
	updateToolbars();
}

void
ViewerWindow::testSlot()
{
	//QImage image = mViewer->RenderThumbnailImage( QSize( 256, 256 ) );
	//label->setPixmap( QPixmap::fromImage( image ) );
}


void
ViewerWindow::applyTransferFunction()
{
	mViewer->setTransferFunctionBuffer( mTransferFunctionEditor->GetTransferFunctionBuffer() );
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
	int viewType = mViewer->getViewType();

	switch ( viewType ) {
	case M4D::GUI::Viewer::vt2DAlignedSlices:
		action2D->setChecked( true );
		actionUse_WLWindow->setEnabled( true );
		actionUse_Simple_Colormap->setEnabled( true );
		actionUse_MIP->setEnabled( false );
		break;
	case M4D::GUI::Viewer::vt2DGeneralSlices:
		ASSERT( false );
		break;
	case M4D::GUI::Viewer::vt3D:
		action3D->setChecked( true );
		actionUse_WLWindow->setEnabled( false );
		actionUse_Simple_Colormap->setEnabled( false );
		actionUse_MIP->setEnabled( true );
		break;
	default:
		ASSERT( false );
	};


	int colorTransformType = mViewer->getColorTransformType();

	switch ( colorTransformType ) {
	case M4D::GUI::Renderer::ctLUTWindow:
		actionUse_WLWindow->setChecked( true );
		break;
	case M4D::GUI::Renderer::ctSimpleColorMap:
		actionUse_Simple_Colormap->setChecked( true );
		break;
	case M4D::GUI::Renderer::ctTransferFunction1D:
		actionUse_Transfer_Function->setChecked( true );
		break;
	case M4D::GUI::Renderer::ctMaxIntensityProjection:
		actionUse_MIP->setChecked( true );
		break;
	default:
		ASSERT( false );
	}
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
	

	//mViewer->ZoomFit();
	applyTransferFunction();
}
