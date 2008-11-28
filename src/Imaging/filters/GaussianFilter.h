#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include "Imaging/Convolution.h"
#include "Imaging/filters/ConvolutionFilter.h"
#include <cmath>

namespace M4D
{
namespace Imaging
{
typedef ConvolutionMask< 2, float32 > ConvolutionMask2DFloat;

ConvolutionMask2DFloat::Ptr
CreateGaussianFilterMask( uint32 radius )
{
	uint32 pom[2];
	float32 std = static_cast<float32>( radius )/3.0;
	float32 *buff = new float32[pom[0]*pom[1]];
	
	double sum = 0.0;
	unsigned idx = 0;
	for( pom[0] = 0; pom[0] < 2*radius + 1; ++pom[0] ) {
		for( pom[1] = 0; pom[1] < 2*radius + 1; ++pom[1] ) {
			sum += buff[ idx ] = 
				exp( - static_cast<float32>( PWR(pom[0]-radius) + PWR(pom[1]-radius) ) / (2.0*PWR(std)) );
		}
	}
	for( idx = 0; idx < PWR( 2*radius + 1 ); ++ idx ) {
		buff[ idx ] /= sum;
	}
	ConvolutionMask2DFloat *maskPtr = new ConvolutionMask< 2, float32 >( buff, pom );

	return ConvolutionMask2DFloat::Ptr( maskPtr );
}


template< typename ImageType >
class GaussianFilter2D :
	public ConvolutionFilter2D< ImageType, float32 >
{
public:
	typedef ConvolutionFilter2D< ImageType, float32 > 	PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType	ElementType;
	typedef	ConvolutionMask2DFloat::Ptr			MaskPtr;
	
	struct Properties : public PredecessorType::Properties
	{
	public:
		uint32	radius;
	protected:
		M4D::Common::TimeStamp	_tmpStamp;
	public:
		Properties()
			{}

		void
		CheckProperties()
			{
				PredecessorType::Properties::CheckProperties();
				if( _tmpStamp != this->GetTimestamp() ) {
					this->matrix = CreateGaussianFilterMask( radius );
				}
			}
	};

	GET_SET_PROPERTY_METHOD_MACRO( uint32, Radius, radius );
private:
	GET_PROPERTIES_DEFINITION_MACRO;
};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*GAUSSIAN_FILTER_H*/
