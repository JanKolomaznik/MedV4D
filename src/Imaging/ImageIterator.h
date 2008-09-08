#ifndef _IMAGE_ITERATOR_H
#define _IMAGE_ITERATOR_H

//#include "Imaging/Image.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageIterator.h 
 * @{ 
 **/

namespace Imaging
{

/*template< typename ElementType, uint32 Dim >
class ImageIterator;

template< typename ElementType >
ImageIterator< ElementType, 2 >
CreateImageIterator(
			ElementType	*pointer, 
			int32		width, 
			int32		height,
			int32		xStride,
			int32		yStride,
			int32		xPos = 0,
			int32		yPos = 0
			);

template< typename ElementType >
ImageIterator< ElementType, 3 >
CreateImageIterator(
			ElementType	*pointer, 
			int32		width, 
			int32		height,
			int32		depth,
			int32		xStride,
			int32		yStride,
			int32		zStride,
			int32		xPos = 0,
			int32		yPos = 0,
			int32		zPos = 0
			);*/


template< typename ElementType, uint32 Dim >
class ImageIterator
{
public:
	static const uint32 Dimension = Dim;

protected:
	ElementType	*_pointer;

	int32	_size[ Dimension ];
	int32	_strides[ Dimension ];
	int32	_position[ Dimension ];
public:
	ImageIterator(): _pointer( NULL ) {}

	ImageIterator( const ImageIterator& it ): _pointer( it._pointer ) 
	{
		for( unsigned i = 0; i < Dimension; ++i ) {
			_position[i] = it._position[i];
			_size[i] = it._size[i];
			_strides[i] = it._strides[i];
			
		}
	}

	ImageIterator( 
			ElementType	*pointer,
			int32		size[], 
			int32		strides[],
			int32		position[]
			): _pointer( pointer ) 
	{
		for( unsigned i = 0; i < Dimension; ++i ) {
			_position[i] = position[i];
			_size[i] = size[i];
			_strides[i] = strides[i];
		}
	}


	ImageIterator&
	operator=( const ImageIterator &it )
	{
		_pointer = it._pointer;
		for( unsigned i = 0; i < Dimension; ++i ) {
			_position[i] = it._position[i];
			_size[i] = it._size[i];
			_strides[i] = it._strides[i];
			
		}
		return *this;
	}

	ImageIterator 
	Begin() const
	{
		ImageIterator result( *this );

		for( unsigned i = 0; i < Dimension; ++i ) {
			result._pointer -= _position[i]*_strides[i];
			result._position[i] = 0;
		}
		return result;
	}

	ImageIterator 
	End() const
	{
		ImageIterator result( *this );

		for( unsigned i = 0; i < Dimension; ++i ) {
			result._pointer += (_size[i]-_position[i]-1)*_strides[i];
			result._position[i] = _size[i] - 1;
		}
		return ++result;
	}

	ElementType& 
	operator*() const
		{
			return *_pointer;
		}

	ElementType *
	operator->() const
		{
			return _pointer;
		}

	ImageIterator &
	operator++()
		{
			for( unsigned i = 0; i < Dimension-1; ++i ) {
				if( _position[i]+1 >= _size[i] ) {
					_position[i] = 0;
				} else {
					++_position[i];
					_pointer += _strides[i];
					return *this;
				}
			}
			++_position[Dimension-1];
			_pointer += _strides[Dimension-1];
			return *this;
		}

	ImageIterator 
	operator++( int )
		{
			ImageIterator pom = *this;
			++(*this);
			return pom;
		}

	ImageIterator &
	operator--()
		{
			for( unsigned i = 0; i < Dimension-1; ++i ) {
				if( _position[i] <= 0 ) {
					_position[i] = _size[i]-1;
				} else {
					--_position[i];
					_pointer -= _strides[i];
					return;
				}
			}
			--_position[Dimension-1];
			_pointer -= _strides[Dimension-1];
			return *this;
		}

	ImageIterator 
	operator--( int )
		{
			ImageIterator pom = *this;
			--(*this);
			return pom;
		}
	
	bool
	operator==( const ImageIterator &it )
		{
			return _pointer == it._pointer;
		}

	bool
	operator!=( const ImageIterator &it )
		{
			return _pointer != it._pointer;
		}
};
;

template< typename ElementType >
ImageIterator< ElementType, 2 >
CreateImageIterator(
			ElementType	*pointer, 
			int32		width, 
			int32		height,
			int32		xStride,
			int32		yStride,
			int32		xPos = 0,
			int32		yPos = 0
			)
{
	int32 _size[2];
	int32 _strides[2];
	int32 _position[2];

	_size[0] = width;
	_size[1] = height;

	_strides[0] = xStride;
	_strides[1] = yStride;

	_position[0] = xPos;
	_position[1] = yPos;

	return ImageIterator< ElementType, 2 >( pointer, _size, _strides, _position );
}

template< typename ElementType >
ImageIterator< ElementType, 3 >
CreateImageIterator(
			ElementType	*pointer, 
			int32		width, 
			int32		height,
			int32		depth,
			int32		xStride,
			int32		yStride,
			int32		zStride,
			int32		xPos = 0,
			int32		yPos = 0,
			int32		zPos = 0
			)
{
	int32 _size[3];
	int32 _strides[3];
	int32 _position[3];

	_size[0] = width;
	_size[1] = height;
	_size[2] = depth;

	_strides[0] = xStride;
	_strides[1] = yStride;
	_strides[2] = zStride;

	_position[0] = xPos;
	_position[1] = yPos;
	_position[2] = zPos;

	return ImageIterator< ElementType, 3 >( pointer, _size, _strides, _position );
}

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_IMAGE_ITERATOR_H*/

