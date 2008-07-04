#ifndef _m4dSelection_h
#error File m4dSelection.tcc cannot be included directly!
#else

#include <math.h>

namespace M4D
{

namespace Selection
{

template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x )
{
    _dim = 1;
    _pointValue = new ElementType[ _dim ];
    _pointValue[0] =  x;
}
    
template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x, ElementType y )
{
    _dim = 2;
    _pointValue = new ElementType[ _dim ];
    _pointValue[0] = x;
    _pointValue[1] = y;
}

template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x, ElementType y, ElementType z )
{
    _dim = 3;
    _pointValue = new ElementType[ _dim ];
    _pointValue[0] = x;
    _pointValue[1] = y;
    _pointValue[2] = z;
}
    
template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t )
{
    _dim = 4;
    _pointValue = new ElementType[ _dim ];
    _pointValue[0] = x;
    _pointValue[1] = y;
    _pointValue[2] = z;
    _pointValue[3] = t;
}

template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( const m4dPoint< ElementType >& m )
{
    _dim = m._dim;
    _pointValue = new ElementType[ _dim ];
    size_t i;
    for ( i = 0; i < _dim; ++i ) _pointValue[i] = m._pointValue[i];
}

template< typename ElementType >
m4dPoint< ElementType >::~m4dPoint()
{
    delete[] _pointValue;
}

template< typename ElementType >
m4dPoint< ElementType >&
m4dPoint< ElementType >::operator=( const m4dPoint< ElementType >& m )
{
    _dim = m._dim;
    _pointValue = new ElementType[ _dim ];
    size_t i;
    for ( i = 0; i < _dim; ++i ) _pointValue[i] = m._pointValue[i];
}

template< typename ElementType >
size_t
m4dPoint< ElementType >::getDimension()
{
    return _dim;
}
    
template< typename ElementType >
const ElementType*
m4dPoint< ElementType >::getAllValues() const
{
    return _pointValue;
}
    
template< typename ElementType >
void
m4dPoint< ElementType >::setAllValues( ElementType* v, size_t dim )
{
    ElementType* tmp = new ElementType[ dim ];
    size_t i;
    for ( i = 0; i < dim; i++ ) tmp[i] = v[i];
    v = _pointValue;
    _pointValue = tmp;
    _dim = dim;
    delete[] v;
}


template< typename ElementType >
const ElementType&
m4dPoint< ElementType >::getParticularValue( size_t i ) const
{
    return _pointValue[i];
}

    
template< typename ElementType >
ElementType&
m4dPoint< ElementType >::getParticularValue( size_t i )
{
    return _pointValue[i];
}

template< typename ElementType >
float
m4dPoint< ElementType >::distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 )
{
    if ( p1.getDimension() != p2.getDimension() ) throw ErrorHandling::ExceptionBase( "Dimension mismatch." );

    size_t i;
    float dist = 0;
    for ( i = 0; i < p1.getDimension(); ++i ) dist += ( p1.getParticularValue( i ) - p2.getParticularValue( i ) ) * ( p1.getParticularValue( i ) - p2.getParticularValue( i ) );
    return sqrt( dist );
}

template< typename ElementType >
m4dShape< ElementType >::m4dShape()
{
    _closed = false;
}

template< typename ElementType >
void
m4dShape< ElementType >::addPoint( m4dPoint< ElementType >& p )
{
    _shapePoints.push_back( p );
}

template< typename ElementType >
void
m4dShape< ElementType >::addAllPoints( m4dShape< ElementType >& s )
{
    for ( typename std::list< m4dPoint< ElementType > >::iterator it = s._shapePoints.begin(); it != s._shapePoints.end(); ++it )
        _shapePoints.push_back( *it );
}

template< typename ElementType >
const std::list< m4dPoint< ElementType > >&
m4dShape< ElementType >::shapeElements() const
{
    return _shapePoints;
}

template< typename ElementType >
std::list< m4dPoint< ElementType > >&
m4dShape< ElementType >::shapeElements()
{
    return _shapePoints;
}

template< typename ElementType >
void
m4dShape< ElementType >::clear()
{
    _shapePoints.clear();
}

template< typename ElementType >
void
m4dShape< ElementType >::deleteLast()
{
    _shapePoints.pop_back();
}

template< typename ElementType >
void
m4dShape< ElementType >::closeShape()
{
    _closed = true;
}

template< typename ElementType >
void
m4dShape< ElementType >::openShape()
{
    _closed = false;
}

template< typename ElementType >
bool
m4dShape< ElementType >::shapeClosed()
{
    return _closed;
}

}/* namespace Selection */
}/* namespace M4D	*/

#endif /* m4dSelection.tcc */
