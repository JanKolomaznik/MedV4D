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
	ConvolutionFilterFtor( const ConvolutionMask<2, float32> &mask ) : _mask( mask ), _size( mask.size ), _center( mask.center ), _lastRow( TypeTraits< int32 >::Min )
	{ 
		_array = new ElType[ _size[0]*_size[1] ]; 
		_leftCorner = Vector< int32, 2 >( -_center[0], -_center[1] );
		_rightCorner = Vector< int32, 2 >( _size[0] - _center[0] - 1, _size[1] - _center[1] - 1 );
	}

	ConvolutionFilterFtor( const ConvolutionFilterFtor &ftor ) : _mask( ftor._mask ), _size( ftor._size ), _lastRow( TypeTraits< int32 >::Min )
	{ 
		_array = new ElType[ _size[0]*_size[1] ]; 
		_leftCorner = Vector< int32, 2 >( -_center[0], -_center[1] );
		_rightCorner = Vector< int32, 2 >( _size[0] - _center[0] - 1, _size[1] - _center[1] - 1 );
	}

	~ConvolutionFilterFtor()
	{ delete [] _array; }

	template< typename Accessor >
	ElType
	Apply( const Vector< int32, 2 > &pos, Accessor &accessor ) 
	{
		Vector< int32, 2 > idx;
		unsigned i = 0;
		if( _lastRow == pos[1] ) {
			idx[0] = pos[0] + _rightCorner[0];
			for( idx[1] = pos[1] + _leftCorner[1]; idx[1] <= pos[1] + _rightCorner[1]; ++idx[1] ) {
				_array[ i*_size[0] + _lastCol ] = accessor( idx );
				++i;
			}
			_lastCol = (_lastCol + 1) % _size[0];
		} else {
			for( idx[1] = pos[1] + _leftCorner[1]; idx[1] <= pos[1] + _rightCorner[1]; ++idx[1] ) {
				for( idx[0] = pos[0] + _leftCorner[0]; idx[0] <= pos[0] + _rightCorner[0]; ++idx[0] ) {
					_array[i++] = accessor( idx );
				}
			}
			_lastCol = 0;
			_lastRow = pos[1];
		}
		typename TypeTraits< ElType >::SuperiorFloatType result = 0;

		Vector< uint32, 2 > midx;
		for( midx[1] = 0; midx[1] < _size[1]; ++midx[1] ) {
			for( midx[0] = 0; midx[0] < _size[0]; ++midx[0] ) {
				uint32 tmp = midx[1] * _size[0] + ((midx[0] + _lastCol) %_size[0]);
				result += _mask.Get( midx ) * _array[ tmp ];
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
	int32	_lastRow;
	int32	_lastCol;
	ElType	*_array;
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
		/*Compute2DConvolution( 
				inRegion, 
				outRegion, 
				*(GetProperties().matrix), 
				GetProperties().addition, 
				GetProperties().multiplication 
				);*/

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

/*
template< typename InputImageType, typename MatrixElement >
ConvolutionFilter3D< InputImageType, MatrixElement >::Properties
::Properties() : PredecessorType::Properties( 0, 10 ), width( 1 ), height( 1 ), depth( 1 )
{
	matrix = MatrixPtr( new MatrixElement[1] );

	matrix[0] = 1;
}

template< typename InputElementType >
ConvolutionFilter3D< Image< InputElementType, 3 > >
::ConvolutionFilter3D() : public PredecessorType( 0, 15 )
{

}

template< typename InputElementType >
bool
ConvolutionFilter3D< Image< InputElementType, 3 > >
::ProcessSlice(	
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
			int32			x1,	
			int32			y1,	
			int32			x2,	
			int32			y2,	
			int32			slice
		    )
{

}*/

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

