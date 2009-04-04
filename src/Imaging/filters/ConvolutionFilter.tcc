/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.tcc 
 * @{ 
 **/

#ifndef _CONVOLUTION_FILTER_H
#error File ConvolutionFilter.tcc cannot be included directly!
#else

#include "Imaging/FilterComputation.h"
/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ElType >
class ConvolutionFilterFtor: public FilterFunctorBase< ElType >
{
public:
	ConvolutionFilterFtor( const ConvolutionMask<2, float32> &mask ) : _mask( mask ), _size( mask.size ), _center( mask.center )
	{ 
		_leftCorner = Vector< int32, 2 >( -_center[0], -_center[1] );
		_rightCorner = Vector< int32, 2 >( _size[0] - _center[0] - 1, _size[1] - _center[1] - 1 );
	}

	ConvolutionFilterFtor( const ConvolutionFilterFtor &ftor ) : _mask( ftor._mask ), _size( ftor._size )
	{ 
		_leftCorner = Vector< int32, 2 >( -_center[0], -_center[1] );
		_rightCorner = Vector< int32, 2 >( _size[0] - _center[0] - 1, _size[1] - _center[1] - 1 );
	}

	~ConvolutionFilterFtor()
	{}

	template< typename Accessor >
	ElType
	Apply( const Vector< int32, 2 > &pos, Accessor &accessor ) 
	{
		Vector< int32, 2 > idx;
		unsigned i = 0;
		typename TypeTraits< ElType >::SuperiorFloatType result = 0;
		const Vector< int32, 2 > lborder( pos + _leftCorner );
		const Vector< int32, 2 > rborder( pos + _rightCorner );

		for( idx[1] = lborder[1]; idx[1] <= rborder[1]; ++idx[1] ) {
			for( idx[0] = lborder[0]; idx[0] <= rborder[0]; ++idx[0] ) {
				result += _mask.mask[i] * accessor( idx );
			}
		}

		return static_cast<ElType>( result );
	}
	Vector< int32, 2 >
	GetLeftCorner()const
	{ return _leftCorner; }

	Vector< int32, 2 >
	GetRightCorner()const
	{ return _rightCorner; }
protected:
	const ConvolutionMask<2, float32> &_mask;
	Vector< uint32, 2 > _size; 
	Vector< uint32, 2 > _center;

	Vector< int32, 2 > _leftCorner; 
	Vector< int32, 2 > _rightCorner;
};


template< typename ImageType, typename MatrixElement >
ConvolutionFilter2D< ImageType, MatrixElement >
::ConvolutionFilter2D() : PredecessorType( new Properties() )
{
	this->_name = "ConvolutionFilter2D";
}

template< typename ImageType, typename MatrixElement >
ConvolutionFilter2D< ImageType, MatrixElement >
::ConvolutionFilter2D( typename ConvolutionFilter2D< ImageType, MatrixElement >::Properties *prop ) 
: PredecessorType( prop ) 
{
	this->_name = "ConvolutionFilter2D";
}

template< typename ImageType, typename MatrixElement >
bool
ConvolutionFilter2D< ImageType, MatrixElement >
::Process2D(
		const typename ConvolutionFilter2D< ImageType, MatrixElement >::Region	&inRegion,
		typename ConvolutionFilter2D< ImageType, MatrixElement >::Region 	&outRegion
		)
{
	try {
		ConvolutionFilterFtor< ElementType > filter( *GetConvolutionMask() );
		FilterProcessorNeighborhood< 
			ConvolutionFilterFtor< ElementType >,
			Region,
			Region,
			MirrorAccessor
			>( filter, inRegion, outRegion );	
	}
	catch( ... ) { 
		return false; 
	}

	return true;
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

