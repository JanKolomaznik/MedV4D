#ifndef _IMAGE_ITERATOR_H
#define _IMAGE_ITERATOR_H

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

template< typename ElementType, uint32 Dim >
class ImageIterator
{
public:
	static const uint32 Dimension = Dim;

protected:
	ElementType	*_pointer;

	uint32	_size[ Dimension ];
	int32	_strides[ Dimension ];
	int32	_contStrides[ Dimension ];
	int32	_position[ Dimension ];
public:
	ImageIterator(): _pointer( NULL ) {}

	ImageIterator( const ImageIterator& it ): _pointer( it._pointer ) 
	{
		for( unsigned i = 0; i < Dimension; ++i ) {
			_position[i] = it._position[i];
			_size[i] = it._size[i];
			_strides[i] = it._strides[i];
			_contStrides[i] = it._contStrides[i];
			
		}
	}

	ImageIterator( 
			ElementType		*pointer,
			const uint32		size[], 
			const int32		strides[],
			const uint32		position[]
			): _pointer( pointer ) 
	{
		for( unsigned i = 0; i < Dimension; ++i ) {
			_position[i] = position[i];
			_size[i] = size[i];
			_strides[i] = strides[i];
			_contStrides[i] = _strides[i];
			for( unsigned j = 0; j < i; ++j ) {
				_contStrides[i] -= (_size[j]-1) * _strides[j];
			}
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
			_contStrides[i] = it._contStrides[i];
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

	bool
	IsEnd()const
	{
		//TODO improve
		return _position[ Dimension-1 ] >= (int32)_size[ Dimension-1 ];
	}

	ElementType *
	GetPointer() const
		{
			return _pointer;
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
				if( _position[i]+1 >= (int32)_size[i] ) {
					_position[i] = 0;
				} else {
					++_position[i];
					_pointer += _contStrides[i];
					return *this;
				}
			}
			++_position[Dimension-1];
			_pointer += _contStrides[Dimension-1];
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
					_pointer -= _contStrides[i];
					return;
				}
			}
			--_position[Dimension-1];
			_pointer -= _contStrides[Dimension-1];
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

template< typename ElementType, unsigned Dim >
ImageIterator< ElementType, Dim >
CreateImageIteratorGen(
			ElementType	*pointer, 
			Vector< uint32, Dim > size,
			Vector< int32, Dim > strides,	
			Vector< uint32, Dim > pos
			)
{
	/*uint32 _size[2];
	int32 _strides[2];
	uint32 _position[2];

	_size[0] = width;
	_size[1] = height;

	_strides[0] = xStride;
	_strides[1] = yStride;

	_position[0] = xPos;
	_position[1] = yPos;*/

	return ImageIterator< ElementType, Dim >( pointer, size.GetData(), strides.GetData(), pos.GetData() );
}

template< typename ElementType >
ImageIterator< ElementType, 2 >
CreateImageIterator(
			ElementType	*pointer, 
			uint32		width, 
			uint32		height,
			int32		xStride,
			int32		yStride,
			int32		xPos = 0,
			int32		yPos = 0
			)
{
	uint32 _size[2];
	int32 _strides[2];
	uint32 _position[2];

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
			uint32		width, 
			uint32		height,
			uint32		depth,
			int32		xStride,
			int32		yStride,
			int32		zStride,
			int32		xPos = 0,
			int32		yPos = 0,
			int32		zPos = 0
			)
{
	uint32 _size[3];
	int32 _strides[3];
	uint32 _position[3];

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

