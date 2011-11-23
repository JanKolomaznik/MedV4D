
#include "MedV4D/Common.h"
#include "Imaging.h"
#include "GUI/m4dGUISliceViewerWidget.h"
#include <iostream>
#include <QApplication>
#include "window.h"


using namespace M4D::Imaging;
using namespace M4D::Viewer;
using namespace std;


typedef Image< uint32, 3 > Image3DType;
typedef ImageConnection< Image3DType > ProducerConn;

int
main( int argc, char** argv )
{

	ProducerConn prodconn( false );

	Image3DType::Ptr inputImage = ImageFactory::CreateEmptyImage3DTyped< uint32 >( 512,512,50 );

	size_t i, j, k;
	uint8* p;
	for ( i = inputImage->GetDimensionExtents(0).minimum; i < inputImage->GetDimensionExtents(2).maximum; ++i )
		for ( j = inputImage->GetDimensionExtents(1).minimum; j < inputImage->GetDimensionExtents(0).maximum; ++j )
			for ( k = inputImage->GetDimensionExtents(2).minimum; k < inputImage->GetDimensionExtents(1).maximum; ++k )
			{
				p = (uint8*) &inputImage->GetElement( j, k, i );// = ( i * j * k ) % 32000;
				p[0] = i * j % 256;
				p[1] = j * k % 256;
				p[2] = i * k % 256;
				p[3] = 0;
			}


	prodconn.PutImage( inputImage );

	QApplication app(argc, argv);
	mywindow mywin( prodconn );
	mywin.show();
	return app.exec();
}
