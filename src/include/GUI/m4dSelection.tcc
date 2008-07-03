#ifndef _m4dSelection_h
#error File m4dSelection.tcc cannot be included directly!
#else

#include <math.h>

namespace M4D
{

namespace Selection
{

template< typename ElementType >
    
template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x )
{
    _dim = 1;
    _pointValue = new ElementType[ _dim ]{ x };
}
    
template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x, ElementType y )
{
    _dim = 2;
    _pointValue = new ElementType[ _dim ]{ x, y };
}

template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x, ElementType y, ElementType z )
{
    _dim = 3;
    _pointValue = new ElementType[ _dim ]{ x, y, z };
}
    
template< typename ElementType >
m4dPoint< ElementType >::m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t )
{
    _dim = 4;
    _pointValue = new ElementType[ _dim ]{ x, y, z, t };
}

size_t
m4dPoint::GetDimension()
{
    return _dim;
}
    
template< typename ElementType >
const ElementType*
m4dSelection< ElementType >::GetAllValues() const
{
    return _pointValue;
}
    
template< typename ElementType >
void
m4dSelection< ElementType >::SetAllValues( ElementType* v, size_t dim )
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
m4dPoint< ElementType >::GetParticularValue( size_t i ) const
{
    return _pointValue[i];
}

    
template< typename ElementType >
ElementType&
m4dPoint< ElementType >::GetParticularValue( size_t i )
{
    return _pointValue[i];
}

template< typename ElementType >
float
m4dPoint< ElementType >::Distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 )
{
    if ( p1.GetDimension() != p2.GetDimension() ) throw ErrorHandling::ExceptionBase( "Dimension mismatch." );

    size_t i;
    float dist = 0;
    for ( i = 0; i < p1.GetDimension(); ++i ) dist += ( p1.GetParticularValue( i ) - p2.GetParticularValue( i ) ) * ( p1.GetParticularValue( i ) - p2.GetParticularValue( i ) );
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
    for ( list< m4dPoint< ElementType > >::iterator i = s._shapePoints.begin(); i != s._shapePoints.end(); ++i )
        _shapePoints.push_back( *i );
}

template< typename ElementType >
const list< m4dPoint< ElementType > >&
m4dShape< ElementType >::ShapeElements() const
{
    return _shapePoints;
}

template< typename ElementType >
list< m4Point< ElementType > >&
m4dShape< ElementType >::ShapeElements()
{
    return _shapePoints;
}

void
m4dShape::closeShape()
{
    _closed = true;
}

void
m4dShape::openShape()
{
    _closed = false;
}

bool
m4dShape::shapeClosed()
{
    return _closed;
}

}/* namespace Selection */
}/* namespace M4D	*/

#endif /* m4dSelection.tcc */
