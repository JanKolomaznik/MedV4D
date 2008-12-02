/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file SobelEdgeDetector.tcc 
 * @{ 
 **/

#ifndef _SOBEL_EDGE_DETECTOR_H
#error File SobelEdgeDetector.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ValueType, typename OutElementType >
struct FirstPassFunctor
{
	void
	operator()( ValueType value, OutElementType & output )
	{
		output = static_cast< OutElementType >( Abs( value ) );
	}
};

template< typename ValueType, typename OutElementType >
struct SecondPassFunctor
{
	void
	operator()( ValueType value, OutElementType & output )
	{
		output = output + static_cast< OutElementType >( Abs( value ) );
	}
};

template< typename ImageType >
SobelEdgeDetector< ImageType >
::SobelEdgeDetector() : PredecessorType( new Properties() )
{
	CreateMatrices();
}

template< typename ImageType >
SobelEdgeDetector< ImageType >
::SobelEdgeDetector( SobelEdgeDetector< ImageType >::Properties *prop ) 
: PredecessorType( prop ) 
{
	CreateMatrices();
}

template< typename ImageType >
bool
SobelEdgeDetector< ImageType >
::Process2D(
		const SobelEdgeDetector< ImageType >::Region	&inRegion,
		SobelEdgeDetector< ImageType >::Region 		&outRegion
		)
{
	try {
		Compute2DConvolutionPostProcess<ElementType, ElementType, float32, 
			FirstPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType > > ( 
				inRegion, 
				outRegion, 
				*xMatrix, 
				0,//TypeTraits< ElementType >::Zero, 
				1.0f,
				FirstPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType >()
				);
		Compute2DConvolutionPostProcess<ElementType, ElementType, float32, 
			SecondPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType > > (  
				inRegion, 
				outRegion, 
				*yMatrix, 
				0,//TypeTraits< ElementType >::Zero, 
				1.0f,
				SecondPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType >()
				);
	}
	catch( ... ) { 
		return false; 
	}

	return true;
}

template< typename ImageType >
void
SobelEdgeDetector< ImageType >
::CreateMatrices()
{
	uint32	size[2];
	size[0] = 3; size[1] = 3;
	float32 *m = new float32[9];
	
	m[0]= -1.0f; m[1]=  0.0f; m[2]=  1.0f;
	m[3]= -2.0f; m[4]=  0.0f; m[5]=  2.0f;
	m[6]= -1.0f; m[7]=  0.0f; m[8]=  1.0f;

	xMatrix = MaskPtr( new Mask( m, size ) );

	m = new float32[9];
	m[0]= -1.0f; m[1]= -2.0f; m[2]= -1.0f;
	m[3]=  0.0f; m[4]=  0.0f; m[5]=  0.0f;
	m[6]=  1.0f; m[7]=  2.0f; m[8]=  1.0f;

	yMatrix = MaskPtr( new Mask( m, size ) );
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_SOBEL_EDGE_DETECTOR_H*/

/** @} */

