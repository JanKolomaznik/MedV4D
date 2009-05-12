#ifndef CANNY_EDGE_DETECTION_H
#define CANNY_EDGE_DETECTION_H

#include "common/Direction.h"

template< typename T >
struct Gradient
{
	T		intensity;
	Direction	direction;
	int		helper;
};

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

		float32 dx = accessor( xIdx2 ) - accessor( xIdx1 );
		float32 dy = accessor( yIdx2 ) - accessor( yIdx1 );

		Gradient<float32> result;
		result.intensity = Sqr( dx ) + Sqr( dy );
		result.direction = 0; //TODO
		result.helper = 0;

		return result;
	}
	Vector< int32, 2 >
	GetLeftCorner()const
	{ return Vector< int32, 2 >( -1, -1 ); }

	Vector< int32, 2 >
	GetRightCorner()const
	{ return Vector< int32, 2 >( 1, 1 ) }
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
	FilterProcessorNeighborhood( filter, smoothInput, gradient );
}

template< typename ElementType >
void
CannyEdgeDetection( 
		const ImageRegion< ElementType, 2 >	&smoothInput,
		ImageRegion< ElementType, 2 >		&output,
		ImageRegion< Gradient<float32>, 2 >	&gradient
		)
{
	ComputeQuantizedGradient( smoothInput, gradient );

	//NonmaximumSuppression( gradient );

	//Hysteresis( );
	
}


#endif /*CANNY_EDGE_DETECTION_H*/
