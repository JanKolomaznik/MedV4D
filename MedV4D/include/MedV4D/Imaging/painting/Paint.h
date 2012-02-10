#include "MedV4D/Imaging/Image.h"




template< typename TImage, typename TBrush >
void
paintLineWithBrush( typename TImage &aImage, Vector<float,TImage::Dimension> aStartPoint, Vector<float,TImage::Dimension> aEndPoint, TBrush &aBrush )
{
	typename TImage::ElementExtentsType eExtents = aImage.GetElementExtents();
	typename Vector<float,TImage::Dimension> Vect;
	
	Vect dir = aEndPoint - aStartPoint;
	float dst = VectorSize( dir );
	VectorNormalization( dir );
	
	float delta = M4D::min( eExtents );
	Vect current = aStartPoint;
	aBrush.apply( aStartPoint );
	for( size_t i = 1; i < dst / delta; ++i ) {
		
		aBrush.apply( current );
		current += dir * delta;
	}
	aBrush.apply( aEndPoint );
	
}


template< typename TImage >
void
paintLineWithBrush( typename TImage &aImage, Vector<float,TImage::Dimension> aStartPoint, Vector<float,TImage::Dimension> aEndPoint, float aRadius, CartesianPlanes aPlane )
{

	
}