#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include "filters/ConvolutionFilter.h"


namespace M4D
{
namespace Imaging
{

ConvolutionMask< float32, 2 >::Ptr
CreateGaussianFilterMask( uint32 radius )
{
	uint32 pom[2] = { 1 + 2*radius };
	float32 *buff = new float32[pom[0]*pom[1]];
	ConvolutionMask< float32, 2 > *maskPtr = new ConvolutionMask< float32, 2 >( buff, pom );

	for( pom[0] = 0; pom[0] <= radius; ++pom[0] ) {
		for( pom[1] = 0; pom[1] <= radius; ++pom[0] ) {
		

		}
	}
	uint32 pom2[2] = { 0 };
	uint32 pom3[2] = { 0 };
	for( pom[0] = 0; pom[0] < radius; ++pom[0] ) {
		pom3[0] = radius+1; pom3[1] = pom[1]+radius+1;
		pom2[0] = radius+1; pom2[1] = pom[1];
		maskPtr->GetElement( pom3 ) = maskPtr->GetElement( pom2 );
		for( pom[1] = 0; pom[1] < radius; ++pom[0] ) {
			pom2[0] = pom[0]+radius+1; pom2[1] = pom[1];
			maskPtr->GetElement( pom ) = maskPtr->GetElement( pom2 );

			pom2[0] = pom[0]+radius+1; pom2[1] = pom[1]+radius+1;
			maskPtr->GetElement( pom ) = maskPtr->GetElement( pom2 );

			pom2[0] = pom[0]; pom2[1] = pom[1]+radius+1;
			maskPtr->GetElement( pom ) = maskPtr->GetElement( pom2 );
		}
	}
}


template< typename ImageType >
class GaussianFilter2D :
	public ConvolutionFilter2D< ImageType, float32 >
{
public:
	typedef ConvolutionFilter2D< ImageType, float32 > PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType	ElementType;		
	
	struct Properties : public PredecessorType::Properties
	{
		Properties()
			{}

		uint32	radius;

	};
private:
	GET_PROPERTIES_DEFINITION_MACRO;
}

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*GAUSSIAN_FILTER_H*/
