//#include "Imaging/ImageFactory.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include "GUI/widgets/Simple2DViewer.h"
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include <QtGui/QWidget>
#include <QtGui/QHBoxLayout>
#include "GUI/widgets/GLThreadedWidget.h"
#include "Imaging/Imaging.h"
#include "common/Common.h"

#ifdef Q_WS_X11
#include <X11/Xlib.h>  // for XInitThreads() call
#endif

using namespace M4D::Imaging;

typedef M4D::Imaging::Image< uint16, 2 > ImageType;
typedef M4D::Imaging::Image< uint8, 2 > MaskType;

//typedef Simple2DViewer<QGLWidget> Viewer;
typedef Simple2DViewer<GLThreadedWidget> Viewer;
//typedef GLThreadedWidget		 Viewer;

class ViewerWindow : public QWidget
{
private:
	Viewer *viewerWidget;
	ImageType::Ptr image;
	MaskType::Ptr mask;
public:
	ViewerWindow();
	~ViewerWindow();
};


ViewerWindow::ViewerWindow(): QWidget( NULL )
{
	viewerWidget = new Viewer( NULL );

	unsigned size = 512;
	image =	M4D::Imaging::ImageFactory::CreateEmptyImage2DTyped< uint16 >( size, size );
	for( unsigned i=0; i <512; ++i ) {
		for( unsigned j=0; j <512; ++j ) {
			image->GetElement( Vector< int32, 2 >( i, j ) ) = (/*(i + j) ^ */i*j*300 )% 4000;

		}
	}

	//ImageRegion< uint8, 2 > *pom = new ImageRegion< uint8, 2 >( image->GetSubRegion( Vector<int32,2>( 128,128 ), Vector<int32,2>( 460,460 ) ) );
	ImageRegion< uint16, 2 > *pom = new ImageRegion< uint16, 2 >( image->GetRegion() );
	viewerWidget->SetImageRegion( pom );
	
	mask =	M4D::Imaging::ImageFactory::CreateEmptyImage2DTyped< uint8 >( size, size );
	for( unsigned i=0; i <512; ++i ) {
		for( unsigned j=0; j <512; ++j ) {
			if( Sqr(i - size/2)+ Sqr(j - size/2) < 3000 ) {
				mask->GetElement( Vector< int32, 2 >( i, j ) ) = 255;
				//std::cout << 1;
			} else {
				mask->GetElement( Vector< int32, 2 >( i, j ) ) = 0;
			}

		}
	}
	ImageRegion< uint8, 2 > *pom2 = new ImageRegion< uint8, 2 >( mask->GetRegion() );
	viewerWidget->SetMaskRegion( pom2 );

	
	QHBoxLayout *mainLayout = new QHBoxLayout;
	mainLayout->addWidget(viewerWidget);
	setLayout(mainLayout);
	resize(800,800);

}

ViewerWindow::~ViewerWindow()
{}

int
main( int argc, char** argv )
{
#ifdef Q_WS_X11
    // this needs to be the first in the app to make Xlib calls thread save
    // needed for OpenGl rendering threads
    XInitThreads();
#endif

	/*std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );*/

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	
	QApplication app(argc, argv);
	ViewerWindow viewer;
	LOG( "Startuji..." );
	viewer.show();
	return app.exec();
}

