#include "ViewerWindow.hpp"
#include "GUI/utils/TransferFunctionBuffer.h"
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


}

ViewerWindow::~ViewerWindow()
{

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
