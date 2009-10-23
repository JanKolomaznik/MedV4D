#ifndef _IMAGE_REGION_H
#define _IMAGE_REGION_H

#include "common/Common.h"
#include "common/Vector.h"
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


class AImageRegion
{
public:
	virtual ~AImageRegion() {}

	virtual uint32 
	GetDimension()const = 0;

	virtual int16 
	GetElementTypeID()const = 0;
};

template< typename EType, uint32 Dim >
class ImageRegion: public AImageRegion
{
public:
	static const uint32 Dimension = Dim;
	typedef EType					ElementType;
	typedef ImageIterator< ElementType, Dim >	Iterator;
	typedef Vector< int32, Dim >			PointType;

	ImageRegion():
			_pointer( NULL ), _origin( 0 ), _elementExtents( 1.0f ), _sourceDimension( 0 ), _pointerCoordinatesInSource( NULL )
		{
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = 0;
				_strides[i] = 0;
				_dimOrder[i] = 0;
			}
			_startPointer = NULL;
		}

	ImageRegion( 
			ElementType 			*pointer, 
			Vector< uint32, Dimension >	size,
			Vector< int32, Dimension >	origin
		) 
		{
			//TODO - check
			_pointer = pointer;
			_sourceDimension = Dimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	
			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = 0;
			}
			_size = size;
			_origin = origin;
			_strides = StridesFromSize( size );
			_elementExtents = Vector< float32, Dimension >( 1.0 );
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_dimOrder[i] = i;
			}
			_startPointer = _pointer - _origin * _strides;
		}

	ImageRegion( 
			ElementType 			*pointer, 
			Vector< uint32, Dimension >	size,
			Vector< int32, Dimension >	strides,
			Vector< float32, Dimension >	elementExtents,
			Vector< uint32, Dimension >	dimOrder,
			uint32 				sourceDimension, 
			const int32*			pointerCoordinatesInSource 
		) 
		{
			_pointer = pointer;
			_sourceDimension = sourceDimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	
			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = pointerCoordinatesInSource[i];
			}
			_size = size;
			_strides = strides;
			_elementExtents = elementExtents;
			_dimOrder = dimOrder;
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_origin[i] = _pointerCoordinatesInSource[ _dimOrder[i] ];
			}
			_startPointer = _pointer - _origin * _strides;
		}

	ImageRegion( const ImageRegion& region )
		{
			_pointer = region._pointer;
			_startPointer = region._startPointer;
			_sourceDimension = region._sourceDimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	_origin = region._origin;
			_size = region._size;
			_strides = region._strides;
			_elementExtents = region._elementExtents;
			_dimOrder = region._dimOrder;

			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = region._pointerCoordinatesInSource[i];
			}
		}

	~ImageRegion()
		{
			if( _pointerCoordinatesInSource ) {
				delete [] _pointerCoordinatesInSource;
			}
		}
	
	/**
	 * Method for obtaining iterator, which can iterate over all elements in this region.
	 * @return Image iterator.
	 **/
	Iterator
	GetIterator()const
		{
			return Iterator( _pointer, GetMinimum(), GetMaximum(), _strides, GetMinimum() );
			//return Iterator( _pointer, _size.GetData(), _strides.GetData(), pos );
		}

	/**
	 * Method for obtaining iterator, which can iterate over elements from bounding box defined by two corners.
	 * @return Image iterator.
	 **/
	Iterator
	GetIterator( const PointType &firstCorner, const PointType &secondCorner )const
		{
			return Iterator( _startPointer + firstCorner*_strides, firstCorner, secondCorner, _strides, firstCorner );
		}

	Iterator
	GetIteratorRel( const PointType &firstCorner, const PointType &secondCorner )const
		{
			//TODO check extents
			return GetIterator( firstCorner + _origin, secondCorner + _origin );
		}

	ElementType *
	GetPointer()const
		{
			return _pointer;
		}

	ElementType *
	GetPointer( const PointType &coords )const
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

	PointType
	GetMinimum()const
		{
			return _origin;
		}

	int32
	GetMaximum( unsigned dim )const
		{
			return _origin[dim] + _size[dim];
		}

	PointType
	GetMaximum()const
		{
			return _origin + PointType( (int32*)_size.GetData() );
		}

	int32
	GetStride( unsigned dim )const
		{
			return _strides[dim];
		}

	PointType
	GetStride()const
		{
			return _strides;
		}


	ImageRegion< ElementType, Dimension - 1 >
	GetSlice( int32 sliceCoord, uint32 perpAxis = Dimension - 1 )const
		{
			return GetSliceRel( sliceCoord - _origin[ perpAxis ], perpAxis );
		}

	ImageRegion< ElementType, Dimension - 1 >
	GetSliceRel( int32 sliceCoord, uint32 perpAxis = Dimension - 1 )const
		{
			if( perpAxis >= Dimension ) {
				_THROW_ ErrorHandling::EBadDimension();
			}
			if( sliceCoord < 0 || sliceCoord >= (int32)_size[perpAxis] ) {
				_THROW_	ErrorHandling::EBadParameter( 
						TO_STRING( "Wrong relative 'sliceCoord = " << sliceCoord << "'. Must in interval <0, " << _size[perpAxis] 
							<< ") for dimension index " << perpAxis <<"." )
						);
			}
			ElementType *pointer = _pointer + sliceCoord*_strides[perpAxis];

			int32 *pom = new int32[ _sourceDimension ];
			for( unsigned i=0; i<_sourceDimension; ++i ) {
				pom[i] = _pointerCoordinatesInSource[i];
			}
			pom[ _dimOrder[perpAxis] ] += sliceCoord * Sgn(_strides[perpAxis]);

			Vector<uint32, Dimension-1> size;
			Vector<int32, Dimension-1> strides;
			Vector<float32, Dimension-1> elementExtents;
			Vector<uint32, Dimension-1> dimOrder;

			unsigned j = 0;
			for( unsigned i = 0; i < Dimension; ++i ) {
				if( i != perpAxis ) {
					size[j] = _size[i];
					strides[j] = _strides[i];
					elementExtents[j] = _elementExtents[i];
					dimOrder[j] = _dimOrder[i];
					++j;
				}
			}

			ImageRegion< ElementType, Dimension-1 > result = 
				ImageRegion< ElementType, Dimension-1 >( pointer, size, strides, elementExtents, dimOrder, _sourceDimension, pom );

			delete [] pom;
			return result;
		}

	ImageRegion &
	operator=( const ImageRegion& region )
		{
			_pointer = region._pointer;
			_startPointer = region._startPointer;
			_sourceDimension = region._sourceDimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	_origin = region._origin;
			_size = region._size;
			_strides = region._strides;
			_elementExtents = region._elementExtents;
			_dimOrder = region._dimOrder;

			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = region._pointerCoordinatesInSource[i];
			}
			return *this;
		}

	/*ImageRegion
	Intersection( const ImageRegion & region );

	ImageRegion
	UnionBBox( const ImageRegion & region );*/

	ElementType &
	GetElement( const PointType &coords )
		{ 	
			return GetElementRel( coords - _origin );
		}
	ElementType
	GetElement( const PointType &coords )const
		{
			return GetElementRel( coords - _origin );
		}

	ElementType &
	GetElementRel( const PointType &coords )
		{
			for( unsigned i = 0; i < Dim; ++i ) {
				if( coords[i] < 0 || coords[i] >= (int32)_size[i] ) {
					_THROW_ ErrorHandling::EBadIndex( 
							TO_STRING( "Parameter (relative coordinates) 'coords = [" 
								<< coords << "]' pointing outside of the region. 'size = [" << _size << "]'" )
								);
				}
			}
			return *(_pointer + coords * _strides );
		}
	ElementType
	GetElementRel( const PointType &coords )const
		{
			for( unsigned i = 0; i < Dim; ++i ) {
				if( coords[i] < 0 || coords[i] >= (int32)_size[i] ) {
					_THROW_ ErrorHandling::EBadIndex( 
							TO_STRING( "Parameter (relative coordinates) 'coords = [" 
								<< coords << "]' pointing outside of the region. 'size = [" << _size << "]'" )
							);
				}
			}
			return *(_pointer + coords * _strides );
		}
	ElementType &
	GetElementFast( const PointType &coords )
		{
			return *(_startPointer + coords * _strides );
		}

	ElementType
	GetElementFast( const PointType &coords )const
		{
			return *(_startPointer + coords * _strides );
		}

	ElementType
	GetElementWorldCoords( const Vector< float32, Dimension > &pos )const
	{
		PointType coords;
		for( unsigned i = 0; i < Dim; ++i ) {
			coords[i] = ROUND( pos[i] / _elementExtents[i] );
		}
		return GetElement( coords );
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

	Vector< float32, Dimension >
	GetElementExtents()const
		{ return _elementExtents; }

	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	uint32 
	GetDimension()const
		{ return Dimension; }
protected:
	
private:
	ElementType			*_pointer;
	ElementType			*_startPointer;
	Vector< uint32, Dimension >	_size;
	PointType	_strides;
	PointType	_origin;
	Vector< float32, Dimension >	_elementExtents;

	Vector< uint32, Dimension >	_dimOrder;
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
			Vector< float32, RegDimension >	elementExtents,
			Vector< uint32, RegDimension >	dimOrder,
			Vector< int32, SourceDimension >	pointerCoordinatesInSource
			)
{
	return ImageRegion< ElementType, RegDimension >( 
			pointer, 
			size, 
			strides, 
			elementExtents,
			dimOrder, 
			SourceDimension, 
			pointerCoordinatesInSource.GetData() 
			);
}

//*****************************************************************************
template< typename RegionType, typename Applicator >
Applicator
ForEachInRegion( RegionType &region, Applicator applicator )
{
	typename RegionType::Iterator iterator = region.GetIterator();
	
	return ForEachByIterator( iterator, applicator );
}

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_IMAGE_REGION_H*/
