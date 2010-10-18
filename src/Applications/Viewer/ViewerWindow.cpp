#include "ViewerWindow.hpp"
#include "GUI/utils/ImageDataRenderer.h"
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
	
	QDockWidget * dockwidget = new QDockWidget;
	mTransferFunctionEditor = new M4D::GUI::TransferFunction1DEditor;
	dockwidget->setWidget( mTransferFunctionEditor );

	mTransferFunctionEditor->SetValueInterval( 0.0f, 3000.0f );
	mTransferFunctionEditor->SetMappedValueInterval( 0.0f, 1.0f );
	mTransferFunctionEditor->SetBorderWidth( 5 );
	addDockWidget (Qt::RightDockWidgetArea, dockwidget );

	mViewer->SetLUTWindow( Vector2f( 500.0f,1000.0f ) );

	mTransFuncTimer.setInterval( 500 );
	QObject::connect( &mTransFuncTimer, SIGNAL( timeout() ), this, SLOT( applyTransferFunction() ) );

	QActionGroup *viewerTypeSwitch = new QActionGroup( this );
	QSignalMapper *viewerTypeSwitchSignalMapper = new QSignalMapper( this );
	viewerTypeSwitch->setExclusive( true );
	viewerTypeSwitch->addAction( action2D );
	viewerTypeSwitch->addAction( action3D );
	viewerTypeSwitchSignalMapper->setMapping( action2D, M4D::GUI::rt2DAlignedSlices );
	viewerTypeSwitchSignalMapper->setMapping( action3D, M4D::GUI::rt3D );
	QObject::connect( action2D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( action3D, SIGNAL( triggered() ), viewerTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( viewerTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeViewerType( int ) ) );

	QActionGroup *colorMapTypeSwitch = new QActionGroup( this );
	QSignalMapper *colorMapTypeSwitchSignalMapper = new QSignalMapper( this );
	colorMapTypeSwitch->setExclusive( true );
	colorMapTypeSwitch->addAction( actionUse_WLWindow );
	colorMapTypeSwitch->addAction( actionUse_Transfer_Function );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_WLWindow, M4D::GUI::ctLUTWindow );
	colorMapTypeSwitchSignalMapper->setMapping( actionUse_Transfer_Function, M4D::GUI::ctTransferFunction1D );
	QObject::connect( actionUse_WLWindow, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( actionUse_Transfer_Function, SIGNAL( triggered() ), colorMapTypeSwitchSignalMapper, SLOT( map() ) );
	QObject::connect( colorMapTypeSwitchSignalMapper, SIGNAL( mapped ( int ) ), this, SLOT( changeColorMapType( int ) ) );


	updateToolbars();
}

ViewerWindow::~ViewerWindow()
{

}

void
ViewerWindow::changeViewerType( int aRendererType )
{
	D_PRINT( "Change viewer type" );
	if ( aRendererType == M4D::GUI::rt3D 
		&& mViewer->GetColorTransformType() == M4D::GUI::ctLUTWindow ) 
	{
		mViewer->SetColorTransformType( M4D::GUI::ctTransferFunction1D );
	}

	mViewer->SetRendererType( aRendererType );
	updateToolbars();
}

void
ViewerWindow::changeColorMapType( int aColorMap )
{
	D_PRINT( "Change color map type" );
	mViewer->SetColorTransformType( aColorMap );
	updateToolbars();
}

void
ViewerWindow::applyTransferFunction()
{
	mViewer->SetTransferFunctionBuffer( mTransferFunctionEditor->GetTransferFunctionBuffer() );
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
	int rendererType = mViewer->GetRendererType();

	switch ( rendererType ) {
	case M4D::GUI::rt2DAlignedSlices:
		action2D->setChecked( true );
		actionUse_WLWindow->setEnabled( true );
		break;
	case M4D::GUI::rt3DGeneralSlices:
		ASSERT( false );
		break;
	case M4D::GUI::rt3D:
		action3D->setChecked( true );
		actionUse_WLWindow->setEnabled( false );
		break;
	default:
		ASSERT( false );
	};


	int colorTransformType = mViewer->GetColorTransformType();

	switch ( colorTransformType ) {
	case M4D::GUI::ctLUTWindow:
		actionUse_WLWindow->setChecked( true );
		break;
	case M4D::GUI::ctTransferFunction1D:
		actionUse_Transfer_Function->setChecked( true );
		break;
	case M4D::GUI::ctMaxIntensityProjection:
		ASSERT( false );
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
ViewerWindow::openFile( const QString aPath )
{
	M4D::Imaging::AImage::Ptr image = 
		M4D::Imaging::ImageFactory::LoadDumpedImage( aPath.toStdString() );

	mProdconn.PutDataset( image );
	mProdconn.ConnectConsumer( mViewer->InputPort()[0] );

	mViewer->ZoomFit();

	M4D::GUI::TransferFunctionBuffer1D::Ptr transferFunction = M4D::GUI::TransferFunctionBuffer1D::Ptr( new M4D::GUI::TransferFunctionBuffer1D( 4096, Vector2f( 0.0f, 4095.0f ) ) );

	M4D::GUI::TransferFunctionBuffer1D::Iterator it;
	float r,g,b;
	float step = 1.0f / (3*4096.0f);
	r = g = b = 0.0f;
	for( it = transferFunction->Begin(); it != transferFunction->End(); ++it ) {
		*it = /*RGBAf( 0.0f, 1.0f, 0.0f, 1.0f );*/RGBAf( r, g, b, 1.0f );
		r += step;
		g += sin( step/50.0f ) * 0.5f + 0.5f;
		b += cos( step/40.0f ) * 0.5f + 0.5f;
	}
	mViewer->SetTransferFunctionBuffer( transferFunction );
}
