//#include "Imaging/ImageFactory.h"

#include "common/Common.h"
#include "common/OGLTools.h"
#include "GUI/widgets/GLThreadedWidget.h"
#include <QtGui/QApplication>
//#include <QtOpenGL/QGLWidget>
#include <QtGui/QWidget>
#include <QtGui/QHBoxLayout>
#include "GUI/widgets/SimpleVolumeViewer.h"
#include "Imaging/Imaging.h"
#include "common/Quaternion.h"
#include <iostream>
#include <fstream>
#include <sstream>



#ifdef Q_WS_X11
#include <X11/Xlib.h>  // for XInitThreads() call
#endif

using namespace M4D::Imaging;

typedef int16 ElementType;
typedef M4D::Imaging::Image< ElementType, 3 > ImageType;
typedef M4D::Imaging::Image< uint8, 2 > MaskType;

//typedef SimpleVolumeViewer<GLThreadedWidget> Viewer;
typedef SimpleVolumeViewer<QGLWidget> Viewer;


class ViewerWindow : public QWidget
{
private:
	Viewer *viewerWidget;
	ImageType::Ptr image;
	AImage::Ptr aimage;
	MaskType::Ptr mask;
public:
	ViewerWindow();
	~ViewerWindow();
};


ViewerWindow::ViewerWindow(): QWidget( NULL )
{
	viewerWidget = new Viewer( NULL );


	/*unsigned size = 128;

	unsigned R = 25;
	unsigned r = 12;

	Vector<float,3> center1( 50,50,64 );
	Vector<float,3> center2( 128-50,128-50,64 );
	image =	M4D::Imaging::ImageFactory::CreateEmptyImage3DTyped< ElementType >( size, size, size );
	for( unsigned i=0; i <size; ++i ) {
		for( unsigned j=0; j <size; ++j ) {
			for( unsigned k=0; k <size; ++k ) {
				//if( Sqr(i - size/2)+ Sqr(j - size/2) + Sqr(k - size/2) < 3000 ) {
				//	image->GetElement( Vector< int32, 3 >( i, j, k ) ) = 255;
				//} else {
				//	image->GetElement( Vector< int32, 3 >( i, j, k ) ) = 0;
				//} 

				float result1 = Sqr( R - Sqrt( Sqr((float)(i -center1[0])) + Sqr((float)(j -center1[1]))) ) + Sqr( (float)(k-center1[2]) );
				float result2 = Sqr( R - Sqrt( Sqr((float)(i -center2[0])) + Sqr((float)(j -center2[1]))) ) + Sqr( (float)(k-center2[2]) );
				if( result1 < Sqr( r ) || result2 < Sqr( r ) || (result1 + result2) < Sqr( 2*r ) ){
					image->GetElement( Vector< int32, 3 >( i, j, k ) ) = 255* Sqr((float)r)/Min(result1,result2);
				} else {
					image->GetElement( Vector< int32, 3 >( i, j, k ) ) = 0;
				}
			}
		}
	}

	for( unsigned i=0; i <size; ++i ) {
		image->GetElement( Vector< int32, 3 >( i, 5, 5 ) ) = 2*i;
	}*/
	//ImageRegion< ElementType, 3 > *pom = new ImageRegion< ElementType, 3 >( image->GetRegion() );
	
	aimage = M4D::Imaging::ImageFactory::LoadDumpedImage( "Skull.dump" );
	AImageRegionDim<3> * pom = dynamic_cast< AImageRegionDim<3>* >( aimage->GetAImageRegion() );
	viewerWidget->SetImageRegion( pom );

	
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

	Vector< float, 3 > a( 4.0f, 0.0f, 0.0f );
	Vector< float, 3 > b( 4.0f, 3.0f, 0.0 );
	Ortogonalize( a, b );

	/*float size = VectorSize( a );
	float product = a * b;
	b -= product * a ;
	LOG( size );
	LOG( product );*/
	LOG( a );
	LOG( b );

	

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	
	QApplication app(argc, argv);
	ViewerWindow viewer;
	LOG( "Startuji..." );
	viewer.show();
	return app.exec();
}

