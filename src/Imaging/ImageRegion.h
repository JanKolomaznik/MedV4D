#ifndef _IMAGE_REGION_H
#define _IMAGE_REGION_H

#include "common/Common.h"
#include "common/Vector.h"
#include "Imaging/AImageRegion.h"
#include "Imaging/DatasetDefinitionTools.h"
#include "Imaging/ImageIterator.h"
#include <memory>

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



template< typename EType, unsigned Dim >
class ImageRegion: public AImageRegionDim< Dim >
{
public:

	STANDARD_DECLARATIONS_MACRO( ImageRegion< EType, Dim > )

	static const unsigned 				Dimension = Dim;
	typedef EType					ElementType;
	typedef EType					Element;
	typedef ImageIterator< ElementType, Dim >	Iterator;
	typedef Vector< int, Dim >			PointType;
	typedef Vector< float, Dim >		ExtentType; // typedefs for gcc4.2 error that cannot parse these templates as default parameters
	typedef Vector< unsigned, Dim >		SizeType;

	CONFIGURABLE_PREPARE_CAST_METHODS_MACRO( Cast, typename ThisClass, AImageRegion );

	ImageRegion():
			AImageRegionDim< Dim >(), _pointer( NULL ), _sourceDimension( 0 ), _pointerCoordinatesInSource( NULL )
		{
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_strides[i] = 0;
				_dimOrder[i] = 0;
			}
			_startPointer = NULL;
		}

	ImageRegion( 
			ElementType 				*pointer, 
			const Vector< unsigned, Dimension >	&size,
			const Vector< int, Dimension >	&origin,
			const ExtentType	&elementExtents = ExtentType( 1.0f )
		) :AImageRegionDim< Dim >( size, origin, elementExtents )
		{
			//TODO - check
			_pointer = pointer;
			_sourceDimension = Dimension;
			_pointerCoordinatesInSource = new int32[_sourceDimension];
		       	
			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = 0;
			}
			_strides = StridesFromSize( size );
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_dimOrder[i] = i;
			}
			_startPointer = _pointer - this->_origin * _strides;
		}

	ImageRegion( 
			ElementType 			*pointer, 
			Vector< unsigned, Dimension >	size,
			Vector< int, Dimension >	strides,
			Vector< float, Dimension >	elementExtents,
			Vector< unsigned, Dimension >	dimOrder,
			unsigned			sourceDimension, 
			const int*			pointerCoordinatesInSource 
		) :AImageRegionDim< Dim >( size, ExtentType( 1.0f ), elementExtents )
		{
			_pointer = pointer;
			_sourceDimension = sourceDimension;
			_pointerCoordinatesInSource = new int[_sourceDimension];
		       	
			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				_pointerCoordinatesInSource[i] = pointerCoordinatesInSource[i];
			}
			_strides = strides;
			_dimOrder = dimOrder;
			for ( unsigned i = 0; i < Dimension; ++i ) {
				this->_origin[i] = _pointerCoordinatesInSource[ _dimOrder[i] ];
			}
			_startPointer = _pointer - this->_origin * _strides;
		}

	ImageRegion( const ImageRegion& region )
		: AImageRegionDim< Dim >( region ), 
			_pointer( region._pointer ),
			_startPointer( region._startPointer ),
			_strides( region._strides ),
			_dimOrder( region._dimOrder ),
			_sourceDimension( region._sourceDimension )
		{
			_pointerCoordinatesInSource = new int[_sourceDimension];
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

	AImageRegion::Ptr
	Clone()
	{
		ImageRegion< EType, Dim > *copy = new ImageRegion< EType, Dim >( *this );
		return AImageRegion::Ptr( copy );
	}

	AImageRegion::ConstPtr
	Clone()const
	{
		ImageRegion< EType, Dim > *copy = new ImageRegion< EType, Dim >( *this );
		return AImageRegion::ConstPtr( copy );
	}
	
	/**
	 * Method for obtaining iterator, which can iterate over all elements in this region.
	 * @return Image iterator.
	 **/
	Iterator
	GetIterator()const
		{
			return Iterator( _pointer, this->GetMinimum(), this->GetMaximum(), _strides, this->GetMinimum() );
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
			return GetIterator( firstCorner + this->_origin, secondCorner + this->_origin );
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

	int
	GetStride( unsigned dim )const
		{
			return _strides[dim];
		}

	PointType
	GetStride()const
		{
			return _strides;
		}


	ImageRegion< ElementType, Dimension >
	GetSubRegion( const PointType &min, const PointType &max )const
		{
			return GetSubRegionRel( min - this->_origin, max - this->_origin );
		}

	/*ImageRegion< ElementType, Dimension >
	GetSubRegionRel( const PointType &min, const PointType &max )const
		{
			if( !(min >= PointType()) ) { 
				_THROW_ ErrorHandling::EBadParameter( TO_STRING( "Parameter 'min = [" << min << "]' pointing outside of image!" ) ); 
			}
			if( !(max <= this->_size) ) { 
				_THROW_ ErrorHandling::EBadParameter( TO_STRING( "Parameter 'max = [" << max << "]' pointing outside of image!" ) ); 
			}


			ElementType * pointer = _pointer;
			PointType size = max - min;

			pointer += min * _strides;

			int *pointerCoordinatesInSource = new int[_sourceDimension];
		       	
			for ( unsigned i = 0; i < _sourceDimension; ++i ) {
				pointerCoordinatesInSource[i] = _pointerCoordinatesInSource[i];
			}
			for ( unsigned i = 0; i < Dimension; ++i ) {
				this->_origin[i] = _pointerCoordinatesInSource[ _dimOrder[i] ];
			}

			return ImageRegion( 
					pointer, 
					size,
					_strides,
					_elementExtents,
					_dimOrder,
					_sourceDimension, 
					pointerCoordinatesInSource 
				);
		}*/

	ImageRegion< ElementType, Dimension - 1 >
	GetSlice( int32 sliceCoord, unsigned perpAxis = Dimension - 1 )const
		{
			return GetSliceRel( sliceCoord - this->_origin[ perpAxis ], perpAxis );
		}

	ImageRegion< ElementType, Dimension - 1 >
	GetSliceRel( int32 sliceCoord, unsigned perpAxis = Dimension - 1 )const
		{
			if( perpAxis >= Dimension ) {
				_THROW_ ErrorHandling::EBadDimension();
			}
			if( sliceCoord < 0 || sliceCoord >= (int32)this->_size[perpAxis] ) {
				_THROW_	ErrorHandling::EBadParameter( 
						TO_STRING( "Wrong relative 'sliceCoord = " << sliceCoord << "'. Must in interval <0, " << this->_size[perpAxis] 
							<< ") for dimension index " << perpAxis <<"." )
						);
			}
			ElementType *pointer = _pointer + sliceCoord*_strides[perpAxis];

			int32 *pom = new int32[ _sourceDimension ];
			for( unsigned i=0; i<_sourceDimension; ++i ) {
				pom[i] = _pointerCoordinatesInSource[i];
			}
			pom[ _dimOrder[perpAxis] ] += sliceCoord * Sgn(_strides[perpAxis]);

			Vector<unsigned, Dimension-1> size;
			Vector<int, Dimension-1> strides;
			Vector<float, Dimension-1> elementExtents;
			Vector<unsigned, Dimension-1> dimOrder;

			unsigned j = 0;
			for( unsigned i = 0; i < Dimension; ++i ) {
				if( i != perpAxis ) {
					size[j] = this->_size[i];
					strides[j] = _strides[i];
					elementExtents[j] = this->_elementExtents[i];
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
		       	this->_origin = region._origin;
			this->_size = region._size;
			_strides = region._strides;
			this->_elementExtents = region._elementExtents;
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
			return GetElementRel( coords - this->_origin );
		}
	ElementType
	GetElement( const PointType &coords )const
		{
			return GetElementRel( coords - this->_origin );
		}

	ElementType &
	GetElementRel( const PointType &coords )
		{
			for( unsigned i = 0; i < Dim; ++i ) {
				if( coords[i] < 0 || coords[i] >= (int32)this->_size[i] ) {
					_THROW_ ErrorHandling::EBadIndex( 
							TO_STRING( "Parameter (relative coordinates) 'coords = [" 
								<< coords << "]' pointing outside of the region. 'size = [" << this->_size << "]'" )
								);
				}
			}
			return *(_pointer + coords * _strides );
		}
	ElementType
	GetElementRel( const PointType &coords )const
		{
			for( unsigned i = 0; i < Dim; ++i ) {
				if( coords[i] < 0 || coords[i] >= (int32)this->_size[i] ) {
					_THROW_ ErrorHandling::EBadIndex( 
							TO_STRING( "Parameter (relative coordinates) 'coords = [" 
								<< coords << "]' pointing outside of the region. 'size = [" << this->_size << "]'" )
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
	GetElementWorldCoords( const Vector< float, Dimension > &pos )const
	{
		PointType coords;
		for( unsigned i = 0; i < Dim; ++i ) {
			coords[i] = ROUND( pos[i] / this->_elementExtents[i] );
		}
		return GetElement( coords );
	}

	unsigned
	GetSourceDimension()const
		{ return _sourceDimension; }

	unsigned
	GetDimensionOrder( unsigned idx )const
		{ 
			if( idx >= Dimension ) {
				_THROW_ ErrorHandling::EBadIndex( "Bad index to dimension order array!");
			}
			return _dimOrder[idx]; 
		}

	int
	GetPointerSourceCoordinates( unsigned idx )const
		{ 
			if( idx >= _sourceDimension ) {
				_THROW_ ErrorHandling::EBadIndex( "Bad index to pointer source coordinates array!");
			}
			return _pointerCoordinatesInSource[idx]; 
		}

	
	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	
protected:
	
private:
	ElementType			*_pointer;
	ElementType			*_startPointer;
/*	Vector< uint32, Dimension >	_size;
	Vector< float32, Dimension >	_elementExtents;
	PointType			_origin;*/
	PointType			_strides;

	Vector< unsigned, Dimension >	_dimOrder;
	unsigned			_sourceDimension;
	int				*_pointerCoordinatesInSource;
};
//*****************************************************************************

typedef ImageRegion< uint8, 2 >	MaskRegion2D;
typedef ImageRegion< uint8, 3 >	MaskRegion3D;

//*****************************************************************************

template< typename ElementType, unsigned RegDimension, unsigned SourceDimension >
ImageRegion< ElementType, RegDimension >
CreateImageRegion(
			ElementType				*pointer, 
			Vector< unsigned, RegDimension >	size, 
			Vector< int, RegDimension >		strides,
			Vector< float, RegDimension >		elementExtents,
			Vector< unsigned, RegDimension >	dimOrder,
			Vector< int, SourceDimension >		pointerCoordinatesInSource
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
