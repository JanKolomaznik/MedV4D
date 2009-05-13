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

		float32 dx = accessor( pos + xIdx2 ) - accessor( pos + xIdx1 );
		float32 dy = accessor( pos + yIdx2 ) - accessor( pos + yIdx1 );

		Vector< float32, 2 > v( dx, dy );
		
		Gradient<float32> result;
		result.intensity = Sqr( dx ) + Sqr( dy );
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
		Gradient<float32> result = accessor( pos );
		Direction neighbourDir1 = result.direction;
		Direction neighbourDir2 = OppositeDirection( result.direction );
		if( 
			result.intensity > accessor( pos + directionOffset[ neighbourDir1 ] ).intensity
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

template< typename InputType, typename OutputType >
void
CannyEdgeDetection( 
		const ImageRegion< InputType, 2 >	&smoothInput,
		ImageRegion< OutputType, 2 >		&output,
		ImageRegion< Gradient<float32>, 2 >	&gradient
		)
{
	ComputeQuantizedGradient( smoothInput, gradient );

	NonmaximumSuppression( gradient );

	//Hysteresis( );
	typename ImageRegion< Gradient<float32>, 2 >::Iterator gradientIterator = gradient.GetIterator();
	typename ImageRegion< OutputType, 2 >::Iterator outputIterator = output.GetIterator();

	while( !gradientIterator.IsEnd() ) {
	
		if( gradientIterator->helper == cNOEDGE ) {
			*outputIterator = 0;
		} else {
			*outputIterator = gradientIterator->intensity;
		}

		++gradientIterator;
		++outputIterator;
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*CANNY_EDGE_DETECTION_H*/
