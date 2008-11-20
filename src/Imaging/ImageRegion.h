#ifndef _IMAGE_REGION_H
#define _IMAGE_REGION_H

#include "Coordinates.h"
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




template< typename ElementType, uint32 Dim >
class ImageRegion
{
public:
	static const uint32 Dimension = Dim;
	typedef ImageIterator< ElementType, Dim >	Iterator;

	ImageRegion( ElementType *pointer, const uint32 size[ Dimension ], const int32 strides[ Dimension ] )
		{
			_pointer = pointer;
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = size[i];
				_strides[i] = strides[i];
			}
		}

	ImageRegion( const ImageRegion& region )
		{
			_pointer = region._pointer;
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = region._size[i];
				_strides[i] = region._strides[i];
			}
		}

	Iterator
	GetIterator()const
		{
			uint32 pos[Dimension] = { 0 };
			return Iterator( _pointer, _size, _strides, pos );
		}

	ElementType *
	GetPointer()const
		{
			return _pointer;
		}

	uint32
	GetSize( unsigned dim )const
		{
			return _size[dim];
		}

	uint32
	GetSize()const
		{
			return _size;
		}


	uint32
	GetStride( unsigned dim )const
		{
			return _strides[dim];
		}

	uint32
	GetStride()const
		{
			return _strides;
		}

	ImageRegion< ElementType, Dimension - 1 >
	GetSlice( int32 sliceCoord )const
		{
			ElementType *pointer = _pointer + sliceCoord*_strides[Dimension-1];
			return ImageRegion< ElementType, Dimension-1 >( pointer, _size, _strides );
		}

	ImageRegion &
	operator=( const ImageRegion& region )
		{
			_pointer = region._pointer;
			for ( unsigned i = 0; i < Dimension; ++i ) {
				_size[i] = region._size[i];
				_strides[i] = region._strides[i];
			}
		}

	/*ImageRegion
	Intersection( const ImageRegion & region );

	ImageRegion
	UnionBBox( const ImageRegion & region );*/

	ElementType &
	GetElement( const Coordinates< int32, Dim > &coords )
		{ 	ElementType *tmp = _pointer;
			//TODO check coordinates
			for( unsigned i = 0; i < Dim; ++i ) {
				tmp += coords[i] * _strides[i];
			}
			return *tmp;
		}
	ElementType
	GetElement( const Coordinates< int32, Dim > &coords )const
		{ 	ElementType *tmp = _pointer;
			//TODO check coordinates
			for( unsigned i = 0; i < Dim; ++i ) {
				tmp += coords[i] * _strides[i];
			}
			return *tmp;
		}
protected:
	
private:
	ElementType	*_pointer;
	uint32		_size[ Dimension ];
	int32		_strides[ Dimension ];
};

//*****************************************************************************

template< typename ElementType >
ImageRegion< ElementType, 2 >
CreateImageRegion(
			ElementType	*pointer, 
			uint32		width, 
			uint32		height,
			int32		xStride,
			int32		yStride
			)
{
	uint32 _size[2];
	int32 _strides[2];

	_size[0] = width;
	_size[1] = height;

	_strides[0] = xStride;
	_strides[1] = yStride;

	return ImageRegion< ElementType, 2 >( pointer, _size, _strides );
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
			)
{
	uint32 _size[3];
	int32 _strides[3];

	_size[0] = width;
	_size[1] = height;
	_size[2] = depth;

	_strides[0] = xStride;
	_strides[1] = yStride;
	_strides[2] = zStride;

	return ImageRegion< ElementType, 3 >( pointer, _size, _strides );
}

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_IMAGE_REGION_H*/
