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

namespace Imaging {

template< typename ElementType, uint32 Dim >
class ImageIterator
{
public:
        static const uint32 Dimension = Dim;
        typedef Vector< int32, Dim >	PointType;

protected:
        ElementType	*_pointer;

        PointType	_minimum;
        PointType	_maximum;

        PointType	_strides;
        PointType	_contStrides;
        PointType	_position;
        Vector< uint32, Dimension > _size;

public:
        ImageIterator() : _pointer ( NULL ) {}

        ImageIterator ( const ImageIterator& it ) : _pointer ( it._pointer ), _minimum ( it._minimum ),
                        _maximum ( it._maximum ), _strides ( it._strides ), _contStrides ( it._contStrides ),
                        _position ( it._position ), _size ( it._size ) {
                _size = _maximum - _minimum;
        }

        ImageIterator (
                ElementType	*pointer,
                PointType	minimum,
                PointType	maximum,
                PointType	strides,
                PointType	position
        ) : _pointer ( pointer ), _minimum ( minimum ),
                        _maximum ( maximum ), _strides ( strides ), _position ( position ) {
                _size = _maximum - _minimum;
                _contStrides = _strides;
                for ( unsigned i = 0; i < Dimension; ++i ) {
                        for ( unsigned j = 0; j < i; ++j ) {
                                _contStrides[i] -= ( _size[j]-1 ) * _strides[j];
                        }
                }
        }


        /*ImageIterator&
        operator=( const ImageIterator &it )
        {
        	_pointer = it._pointer;
        	_minimum = it._minimum;
        	_maximum = it._maximum;
        	_strides = it._strides;
        	_contStrides = it._contStrides;
        	_position = it._position;
        	_size = it._size;
        	return *this;
        }*/

        ImageIterator
        Begin() const {
                ImageIterator result ( *this );

                result._pointer -= ( _position - _minimum ) * _strides;
                result._position = _minimum;
                return result;
        }

        ImageIterator
        End() const {
                ImageIterator result ( *this );
                PointType one ( 1 );
                result._pointer += ( _maximum - _position - one ) *_strides;
                result._position = _maximum - one;

                return ++result;
        }

        bool
        IsEnd() const {
                //TODO improve
                return _position[ Dimension-1 ] >= _maximum[ Dimension-1 ];
        }

        const PointType &
        GetCoordinates() const {
                return _position;
        }

        ElementType *
        GetPointer() const {
                return _pointer;
        }

        ElementType&
        operator*() const {
                return *_pointer;
        }

        ElementType *
        operator->() const {
                return _pointer;
        }

        ImageIterator
        operator+ ( const Vector<int32, Dimension> &v ) {
                ImageIterator result ( *this );
                result._pointer += v * _strides;
                result._position += v;
                return result;
        }

        ImageIterator
        operator- ( const Vector<int32, Dimension> &v ) {
                ImageIterator result ( *this );
                result._pointer -= v * _strides;
                result._position -= v;
                return result;
        }

        ImageIterator &
        operator++() {
                for ( unsigned i = 0; i < Dimension-1; ++i ) {
                        if ( _position[i]+1 == _maximum[i] ) {
                                _position[i] = _minimum[i];
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
        operator++ ( int ) {
                ImageIterator pom = *this;
                ++ ( *this );
                return pom;
        }

        ImageIterator &
        operator--() {
                for ( unsigned i = 0; i < Dimension-1; ++i ) {
                        if ( _position[i] == _minimum[i] ) {
                                _position[i] = _maximum[i]-1;
                        } else {
                                --_position[i];
                                _pointer -= _contStrides[i];
                                return *this;
                        }
                }
                --_position[Dimension-1];
                _pointer -= _contStrides[Dimension-1];
                return *this;
        }

        ImageIterator
        operator-- ( int ) {
                ImageIterator pom = *this;
                -- ( *this );
                return pom;
        }

        bool
        operator== ( const ImageIterator &it ) {
                return _pointer == it._pointer;
        }

        bool
        operator!= ( const ImageIterator &it ) {
                return _pointer != it._pointer;
        }
};



template< typename IteratorType, typename Applicator >
Applicator
ForEachByIterator ( IteratorType iterator, Applicator applicator )
{
        for ( ; !iterator.IsEnd(); ++iterator ) {
                applicator ( *iterator, iterator.GetCoordinates() );
        }
        return applicator;
}

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_IMAGE_ITERATOR_H*/

