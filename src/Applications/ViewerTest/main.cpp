
#include "Common.h"
#include "Imaging/ExampleImageFilters.h"
#include "Imaging/DefaultConnection.h"
#include "Imaging/ImageFactory.h"
#include "GUI/m4dSliceViewerWidget.h"
#include <iostream>
#include <QApplication>
#include "window.h"


using namespace M4D::Imaging;
using namespace M4D::Viewer;
using namespace std;


typedef Image< int16, 3 > Image3DType;
typedef ImageConnectionSimple< Image3DType > ProducerConn;

int
main( int argc, char** argv )
{

	ProducerConn prodconn;

	Image3DType::Ptr inputImage = ImageFactory::CreateEmptyImage3DTyped< int16 >( 512,512,50 );

	int i, j, k;
	for ( i = inputImage->GetDimensionExtents(0).minimum; i < inputImage->GetDimensionExtents(2).maximum; ++i )
		for ( j = inputImage->GetDimensionExtents(1).minimum; j < inputImage->GetDimensionExtents(0).maximum; ++j )
			for ( k = inputImage->GetDimensionExtents(2).minimum; k < inputImage->GetDimensionExtents(1).maximum; ++k )
				inputImage->GetElement( j, k, i ) = ( i * j * k ) % 32000;


	prodconn.PutImage( inputImage );

	QApplication app(argc, argv);
	mywindow mywin( prodconn );
	mywin.show();
	return app.exec();
}
