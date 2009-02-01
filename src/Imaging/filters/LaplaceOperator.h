#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include "Imaging/Convolution.h"
#include "Imaging/filters/ConvolutionFilter.h"
#include <cmath>

namespace M4D
{
namespace Imaging
{
typedef ConvolutionMask< 2, float32 > ConvolutionMask2DFloat;

inline ConvolutionMask2DFloat::Ptr
CreateLaplacianFilterMask()
{
	uint32	size[2];
	size[0] = 3; size[1] = 3;
	float32 *m = new float32[9];
	
	m[0]=  0.0f; m[1]= -1.0f; m[2]=  0.0f;
	m[3]= -1.0f; m[4]=  4.0f; m[5]= -1.0f;
	m[6]=  0.0f; m[7]= -1.0f; m[8]=  0.0f;

	return ConvolutionMask2DFloat::Ptr( new ConvolutionMask2DFloat(m,size) );
}


template< typename ImageType >
class LaplaceOperator2D :
	public ConvolutionFilter2D< ImageType, float32 >
{
public:
	typedef ConvolutionFilter2D< ImageType, float32 > 	PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType	ElementType;
	typedef	ConvolutionMask2DFloat::Ptr			MaskPtr;
	
	struct Properties : public PredecessorType::Properties
	{
	public:
		Properties()
			{ this->matrix = CreateLaplacianFilterMask(); }

	};


	LaplaceOperator2D( Properties * prop ) :  PredecessorType( prop )
		{ /*empty*/ }
	LaplaceOperator2D() :  PredecessorType( new Properties() )
		{ /*empty*/ }

private:
	GET_PROPERTIES_DEFINITION_MACRO;
};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*LAPLACE_OPERATOR_H*/
