#ifndef _IMAGE_REGION_H
#define _IMAGE_REGION_H

#include "Vector.h"
#include "Imaging/ImageIterator.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageRegion.h 
 * @{ 
 **/

namespace Imaging
{




template< typename EType, uint32 Dim >
class ImageRegion
{
public:
	static const uint32 Dimension = Dim;
	typedef EType					ElementType;
	typedef ImageIterator< ElementType, Dim >	Iterator;

	ImageRegion():
			_pointer( NULL ), _sourceDimension( 0 ), _pointerCoordinatesInSource( NULL ), _origin( 0 )
		{
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = 0;
				_strides[i] = 0;
				_dimOrder[i] = 0;
			}
		}

	ImageRegion( 
			ElementType 	*pointer, 
			const uint32 	size[ Dimension ], 
			const int32 	strides[ Dimension ], 
			const uint32 	dimOrder[ Dimension ], 
			uint32 		sourceDimension, 
			const int32*	pointerCoordinatesInSource 
		): _origin( 0 )
		{
			_pointer = pointer;
			_sourceDimension = sourceDimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	
			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = pointerCoordinatesInSource[i];
			}
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = size[i];
				_strides[i] = strides[i];
				_dimOrder[i] = dimOrder[i];
			}
		}

	ImageRegion( const ImageRegion& region )
		{
			_pointer = region._pointer;
			_sourceDimension = region._sourceDimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	_origin = region._origin;
			_size = region._size;
			_strides = region._strides;

			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = region._pointerCoordinatesInSource[i];
			}
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_dimOrder[i] = region._dimOrder[i];
			}
		}

	~ImageRegion()
		{
			if( _pointerCoordinatesInSource ) {
				delete [] _pointerCoordinatesInSource;
			}
		}

	Iterator
	GetIterator()const
		{
			uint32 pos[Dimension] = { 0 };
			return Iterator( _pointer, _size.GetData(), _strides.GetData(), pos );
		}

	Iterator
	GetIterator( const Vector< int32, Dim > &firstCorner, const Vector< int32, Dim > &secondCorner )const
		{
			return GetIteratorRel( firstCorner - _origin, secondCorner - _origin );
		}

	Iterator
	GetIteratorRel( const Vector< int32, Dim > &firstCorner, const Vector< int32, Dim > &secondCorner )const
		{
			//TODO check extents
			uint32 pos[Dimension] = { 0 };
			uint32 size[Dimension];
			for( unsigned i=0; i<Dimension; ++i )
			{
				size[i] = secondCorner[i]-firstCorner[i];
			}
			return Iterator( &GetElementRel(firstCorner), size, _strides, pos );
		}

	ElementType *
	GetPointer()const
		{
			return _pointer;
		}

	ElementType *
	GetPointer( const Vector< int32, Dim > &coords )const
		{ 	ElementType *tmp = _pointer;
			//TODO check coordinates
			for( unsigned i = 0; i < Dim; ++i ) {
				tmp += coords[i] * _strides[i];
			}
			return tmp;
		}

	uint32
	GetSize( unsigned dim )const
		{
			return _size[dim];
		}

	Vector< uint32, Dimension >
	GetSize()const
		{
			return _size;
		}

	int32
	GetMinimum( unsigned dim )const
		{
			return _origin[dim];
		}

	Vector< int32, Dimension >
	GetMinimum()const
		{
			return _origin;
		}

	int32
	GetMaximum( unsigned dim )const
		{
			return _origin[dim] + _size[dim];
		}

	Vector< int32, Dimension >
	GetMaximum()const
		{
			return _origin + _size;
		}

	int32
	GetStride( unsigned dim )const
		{
			return _strides[dim];
		}

	Vector< int32, Dimension >
	GetStride()const
		{
			return _strides;
		}


	ImageRegion< ElementType, Dimension - 1 >
	GetSlice( int32 sliceCoord )const
		{
			return GetSliceRel( sliceCoord - _origin[ Dimension-1 ] );
		}

	ImageRegion< ElementType, Dimension - 1 >
	GetSliceRel( int32 sliceCoord )const
		{
			ElementType *pointer = _pointer + sliceCoord*_strides[Dimension-1];

			int32 *pom = new int32[ _sourceDimension ];
			for( unsigned i=0; i<_sourceDimension; ++i ) {
				pom[i] = _pointerCoordinatesInSource[i];
			}
			pom[ _dimOrder[Dimension-1] ] += sliceCoord * Sgn(_strides[Dimension-1]);

			ImageRegion< ElementType, Dimension-1 > result = 
				ImageRegion< ElementType, Dimension-1 >( pointer, _size.GetData(), _strides.GetData(), _dimOrder, _sourceDimension, pom );

			delete [] pom;
			return result;
		}

	ImageRegion &
	operator=( const ImageRegion& region )
		{
			_pointer = region._pointer;
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = region._size[i];
				_strides[i] = region._strides[i];
			}
			return *this;
		}

	/*ImageRegion
	Intersection( const ImageRegion & region );

	ImageRegion
	UnionBBox( const ImageRegion & region );*/

	ElementType &
	GetElement( const Vector< int32, Dim > &coords )
		{ 	
			return GetElementRel( coords - _origin );
		}
	ElementType
	GetElement( const Vector< int32, Dim > &coords )const
		{
			return GetElementRel( coords - _origin );
		}

	ElementType &
	GetElementRel( const Vector< int32, Dim > &coords )
		{ 	ElementType *tmp = _pointer;
			for( unsigned i = 0; i < Dim; ++i ) {
				if( coords[i] < 0 || coords[i] >= (int32)_size[i] ) {
					_THROW_ ErrorHandling::EBadIndex( "Bad index to ImageRegion!");
				}
				tmp += coords[i] * _strides[i];
			}
			return *tmp;
		}
	ElementType
	GetElementRel( const Vector< int32, Dim > &coords )const
		{ 	ElementType *tmp = _pointer;
			for( unsigned i = 0; i < Dim; ++i ) {
				if( coords[i] < 0 || coords[i] >= (int32)_size[i] ) {
					_THROW_ ErrorHandling::EBadIndex( "Bad index to ImageRegion!");
				}
				tmp += coords[i] * _strides[i];
			}
			return *tmp;
		}

	uint32
	GetSourceDimension()const
		{ return _sourceDimension; }

	uint32
	GetDimensionOrder( unsigned idx )const
		{ 
			if( idx >= Dimension ) {
				_THROW_ ErrorHandling::EBadIndex( "Bad index to dimension order array!");
			}
			return _dimOrder[idx]; 
		}

	int32
	GetPointerSourceCoordinates( unsigned idx )const
		{ 
			if( idx >= _sourceDimension ) {
				_THROW_ ErrorHandling::EBadIndex( "Bad index to pointer source coordinates array!");
			}
			return _pointerCoordinatesInSource[idx]; 
		}
protected:
	
private:
	ElementType	*_pointer;
	Vector< uint32, Dimension >	_size;
	Vector< int32, Dimension >		_strides;
	Vector< int32, Dimension >		_origin;

	uint32		_dimOrder[ Dimension ];
	uint32		_sourceDimension;
	int32		*_pointerCoordinatesInSource;
};

//*****************************************************************************

template< typename ElementType, unsigned RegDimension, unsigned SourceDimension >
ImageRegion< ElementType, RegDimension >
CreateImageRegion(
			ElementType	*pointer, 
			Vector< uint32, RegDimension >	size, 
			Vector< int32, RegDimension >	strides,
			Vector< uint32, RegDimension >	dimOrder,
			Vector< int32, SourceDimension >	pointerCoordinatesInSource
			)
{
	return ImageRegion< ElementType, RegDimension >( 
			pointer, 
			size.GetData(), 
			strides.GetData(), 
			dimOrder.GetData(), 
			SourceDimension, 
			pointerCoordinatesInSource.GetData() 
			);
}

/*template< typename ElementType >
ImageRegion< ElementType, 2 >
CreateImageRegion(
			ElementType	*pointer, 
			uint32		width, 
			uint32		height,
			int32		xStride,
			int32		yStride,
			uint32		xDimOrder = 0,
			uint32		yDimOrder = 1
			)
{
	uint32 _size[2];
	int32 _strides[2];
	uint32 _dimOrder[2];

	_size[0] = width;
	_size[1] = height;

	_strides[0] = xStride;
	_strides[1] = yStride;

	_dimOrder[0] = xDimOrder;
	_dimOrder[1] = yDimOrder;

	return ImageRegion< ElementType, 2 >( pointer, _size, _strides, _dimOrder );
}

template< typename ElementType >
ImageRegion< ElementType, 3 >
CreateImageRegion(
			ElementType	*pointer, 
			uint32		width, 
			uint32		height,
			uint32		depth,
			int32		xStride,
			int32		yStride,
			int32		zStride
			uint32		xDimOrder = 0,
			uint32		yDimOrder = 1,
			uint32		zDimOrder = 2
			)
{
	uint32 _size[3];
	int32 _strides[3];
	uint32 _dimOrder[3];

	_size[0] = width;
	_size[1] = height;
	_size[2] = depth;

	_strides[0] = xStride;
	_strides[1] = yStride;
	_strides[2] = zStride;

	_dimOrder[0] = xDimOrder;
	_dimOrder[1] = yDimOrder;
	_dimOrder[2] = zDimOrder;

	return ImageRegion< ElementType, 3 >( pointer, _size, _strides, _dimOrder );
}*/

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_IMAGE_REGION_H*/
