#include "ViewerWindow.hpp"
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

	mTransFuncTimer.setInterval( 500 );
	QObject::connect( &mTransFuncTimer, SIGNAL( timeout() ), this, SLOT( applyTransferFunction() ) );

}

ViewerWindow::~ViewerWindow()
{

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
