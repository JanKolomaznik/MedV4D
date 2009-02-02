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
		output = static_cast< OutElementType >( Min( Abs( value ), TypeTraits< OutElementType >::Max ) );
	}
};

template< typename ValueType, typename OutElementType >
struct SecondPassFunctor
{
	void
	operator()( ValueType value, OutElementType & output )
	{
		output = static_cast< OutElementType >( Min( static_cast< ValueType >( output ) + Abs( value ), TypeTraits< OutElementType >::Max ) );
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
::SobelEdgeDetector( typename SobelEdgeDetector< ImageType >::Properties *prop ) 
: PredecessorType( prop ) 
{
	CreateMatrices();
}

template< typename ImageType >
bool
SobelEdgeDetector< ImageType >
::Process2D(
		const typename SobelEdgeDetector< ImageType >::Region	&inRegion,
		typename SobelEdgeDetector< ImageType >::Region 	&outRegion
		)
{
	try {
		Compute2DConvolutionPostProcess<ElementType, ElementType, float32, 
			FirstPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType > > ( 
				inRegion, 
				outRegion, 
				*xMatrix, 
				TypeTraits< ElementType >::Zero, 
				1.0f,
				FirstPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType >()
				);
		Compute2DConvolutionPostProcess<ElementType, ElementType, float32, 
			SecondPassFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, ElementType > > (  
				inRegion, 
				outRegion, 
				*yMatrix, 
				TypeTraits< ElementType >::Zero, 
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
//*************************************************
template< typename ValueType, typename OutElementType >
struct FirstPassGradientFunctor
{
	void
	operator()( ValueType value, SimpleVector< OutElementType, 2 > & output )
	{
		output.data[0] = static_cast< OutElementType >( value );
	}
};

template< typename ValueType, typename OutElementType >
struct SecondPassGradientFunctor
{
	void
	operator()( ValueType value, SimpleVector< OutElementType, 2 > & output )
	{
		output.data[1] = static_cast< OutElementType >( value );
	}
};


template< typename ImageType, typename OutType >
SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >
::SobelGradientOperator() : PredecessorType( new Properties() )
{
	CreateMatrices();
}

template< typename ImageType, typename OutType >
SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >
::SobelGradientOperator( typename SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >::Properties *prop ) 
: PredecessorType( prop ) 
{
	CreateMatrices();
}

template< typename ImageType, typename OutType >
bool
SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >
::Process2D(
		const typename SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >::IRegion	&inRegion,
		typename SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >::ORegion 		&outRegion
		)
{
	try {
		Compute2DConvolutionPostProcess<ElementType, OutElementType, float32, 
			FirstPassGradientFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, OutScalarType > > ( 
				inRegion, 
				outRegion, 
				*xMatrix, 
				TypeTraits< ElementType >::Zero, 
				1.0f,
				FirstPassGradientFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, OutScalarType >()
				); 
		Compute2DConvolutionPostProcess<ElementType, OutElementType, float32, 
			SecondPassGradientFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, OutScalarType > > (  
				inRegion, 
				outRegion, 
				*yMatrix, 
				TypeTraits< ElementType >::Zero, 
				1.0f,
				SecondPassGradientFunctor< typename TypeTraits< ElementType >::SuperiorFloatType, OutScalarType >()
				);
	}
	catch( ... ) { 
		return false; 
	}

	return true;
}

template< typename ImageType, typename OutType >
void
SobelGradientOperator< ImageType, Image< SimpleVector< OutType, 2 >, ImageTraits< ImageType >::Dimension > >
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

