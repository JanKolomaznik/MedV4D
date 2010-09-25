#ifndef CANNY_EDGE_DETECTION_H
#define CANNY_EDGE_DETECTION_H

#include "common/Direction.h"
#include "Imaging/FilterComputation.h"

namespace M4D
{
namespace Imaging
{

template< typename T >
struct Gradient
{
	T		intensity;
	Direction	direction;
	int		helper;
};

enum{ cDEFAULT, cPOSSIBLE_EDGE, cEDGE, cNOEDGE };

class GradientComputationFilter: public FilterFunctorBase< Gradient<float32> >
{
public:
	GradientComputationFilter()
	{}
	~GradientComputationFilter()
	{}

	template< typename Accessor >
	Gradient<float32>
	Apply( const Vector< int32, 2 > &pos, Accessor &accessor ) 
	{
		
		static const Vector< int32, 2 > xIdx1( -1, 0 );
		static const Vector< int32, 2 > xIdx2( 1, 0 );
		static const Vector< int32, 2 > yIdx1( 0, -1 );
		static const Vector< int32, 2 > yIdx2( 0, -1 );

		float32 dx = (float32) ( accessor( pos/* + xIdx2 */) - accessor( pos + xIdx1 ) );
		float32 dy = (float32) ( accessor( pos/* + yIdx2 */) - accessor( pos + yIdx1 ) );

		Vector< float32, 2 > v( dx, dy );
		
		Gradient<float32> result;
		result.intensity = sqrt(Sqr( dx ) + Sqr( dy ));
		//result.intensity = Sqr( dx ) + Sqr( dy );
		//result.intensity = dx + dy;
		result.direction = VectorDirection( v ); //TODO
		result.helper = cDEFAULT;

		return result;
	}
	Vector< int32, 2 >
	GetLeftCorner()const
	{ return Vector< int32, 2 >( -1, -1 ); }

	Vector< int32, 2 >
	GetRightCorner()const
	{ return Vector< int32, 2 >( 1, 1 ); }
protected:
};


class NonmaximumSuppressionFilter: public FilterFunctorBase< Gradient<float32> >
{
public:
	NonmaximumSuppressionFilter()
	{}
	~NonmaximumSuppressionFilter()
	{}

	template< typename Accessor >
	Gradient<float32>
	Apply( const Vector< int32, 2 > &pos, Accessor &accessor ) 
	{
		static const float32 THRESHOLD = 0.5;
		Gradient<float32> result = accessor( pos );
		Direction neighbourDir1 = result.direction;
		Direction neighbourDir2 = OppositeDirection( result.direction );
		if( 
			result.intensity > THRESHOLD
			&& result.intensity > accessor( pos + directionOffset[ neighbourDir1 ] ).intensity
			&& result.intensity > accessor( pos + directionOffset[ neighbourDir2 ] ).intensity )
		{
			result.helper = cPOSSIBLE_EDGE;
		} else {
			result.helper = cNOEDGE;
		}

		return result;
	}
	Vector< int32, 2 >
	GetLeftCorner()const
	{ return Vector< int32, 2 >( -1, -1 ); }

	Vector< int32, 2 >
	GetRightCorner()const
	{ return Vector< int32, 2 >( 1, 1 ); }
protected:
};

template< typename ElementType >
void
ComputeQuantizedGradient(
		const ImageRegion< ElementType, 2 >	&smoothInput,
		ImageRegion< Gradient<float32>, 2 >	&gradient
		)
{
	GradientComputationFilter filter;
	FilterProcessorNeighborhoodSimple( filter, smoothInput, gradient );
}

void
NonmaximumSuppression( ImageRegion< Gradient<float32>, 2 > &gradient )
{
	NonmaximumSuppressionFilter filter;
	FilterProcessorNeighborhoodSimple( filter, gradient, gradient );
}
void
TraceEdge( ImageRegion< Gradient<float32>, 2 >::Iterator gradientIterator, float32 lowThreshold )
{
	for( unsigned i = 0; i < 8; ++i ) {
		ImageRegion< Gradient<float32>, 2 >::Iterator tmpIterator = gradientIterator + directionOffset[i];
		if( tmpIterator->helper == cPOSSIBLE_EDGE 
			&& tmpIterator->intensity > lowThreshold 
		) {
			tmpIterator->helper = cEDGE;
			TraceEdge( tmpIterator, lowThreshold );
		}
	}
}

void
Hysteresis( ImageRegion< Gradient<float32>, 2 > &gradient, float32 lowThreshold, float32 highThreshold )
{
	ImageRegion< Gradient<float32>, 2 >::Iterator gradientIterator = gradient.GetIterator();
	
	static const unsigned int HISTOGRAM_SIZE = 256;
	uint64 histogram[ HISTOGRAM_SIZE+1 ] = {0};

	float32 max = gradientIterator->intensity;
	while( !gradientIterator.IsEnd() ) {
		if( gradientIterator->helper == cPOSSIBLE_EDGE ) {
			if( max < gradientIterator->intensity ) {
				max = gradientIterator->intensity;
			}
		}
		++gradientIterator;
	}
	float32 step = max / HISTOGRAM_SIZE;

	gradientIterator = gradientIterator.Begin();
	while( !gradientIterator.IsEnd() ) {
		if( gradientIterator->helper == cPOSSIBLE_EDGE ) {
			++histogram[ (int)(gradientIterator->intensity / step) ];
		}
		++gradientIterator;
	}
	uint64 count = 0;
	for( unsigned i = 1; i <= HISTOGRAM_SIZE; ++i ) {
		count += histogram[i];
	}
	uint64 t = static_cast<uint64>( highThreshold * count ); 

	float32 tHigh = max * 0.8f;
	count = 0;
	for( unsigned i = 1; i <= HISTOGRAM_SIZE; ++i ) {
		count += histogram[i];
		if( i > 1 && count > t ) {
			tHigh = (i-1) * step;
			break;
		}
	}
	float32 tLow = lowThreshold * tHigh;

	D_PRINT( "Low threshold = " << tLow );
	D_PRINT( "High threshold = " << tHigh );

	gradientIterator = gradientIterator.Begin();
	while( !gradientIterator.IsEnd() ) {
		if( gradientIterator->helper == cPOSSIBLE_EDGE ) {
			if( gradientIterator->intensity > tHigh ) {
				gradientIterator->helper = cEDGE;
				TraceEdge( gradientIterator, tLow ); 
			}
		}
		++gradientIterator;
	}
}

template< typename InputType, typename OutputType >
void
CannyEdgeDetection( 
		const ImageRegion< InputType, 2 >	&smoothInput,
		ImageRegion< OutputType, 2 >		&output,
		ImageRegion< Gradient<float32>, 2 >	&gradient,
		float32					lowThreshold, 
		float32					highThreshold
		)
{
	ComputeQuantizedGradient( smoothInput, gradient );

	NonmaximumSuppression( gradient );

	Hysteresis( gradient, lowThreshold, highThreshold );
	typename ImageRegion< Gradient<float32>, 2 >::Iterator gradientIterator = gradient.GetIterator();
	typename ImageRegion< OutputType, 2 >::Iterator outputIterator = output.GetIterator();

	while( !gradientIterator.IsEnd() ) {
	
		if( gradientIterator->helper != cEDGE ) {
			*outputIterator = 0;
		} else {
			*outputIterator = 255;//gradientIterator->intensity;
		}

		++gradientIterator;
		++outputIterator;
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*CANNY_EDGE_DETECTION_H*/
