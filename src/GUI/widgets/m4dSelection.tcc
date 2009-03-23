/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dSelection.tcc 
 * @{ 
 **/

#ifndef _m4dSelection_h
#error File m4dSelection.tcc cannot be included directly!
#else

#include <math.h>

namespace M4D
{

namespace Selection
{

template< typename ElementType >
m4dPoint< ElementType >::m4dPoint()
{
    _dim = 0;
    _pointValue = 0;
}

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
    if ( _dim > 0 )
    {
        _pointValue = new ElementType[ _dim ];
        size_t i;
        for ( i = 0; i < _dim; ++i ) _pointValue[i] = m._pointValue[i];
    }
    else _pointValue = 0;
}

template< typename ElementType >
m4dPoint< ElementType >::~m4dPoint()
{
    if ( _pointValue ) delete[] _pointValue;
}

template< typename ElementType >
m4dPoint< ElementType >&
m4dPoint< ElementType >::operator=( const m4dPoint< ElementType >& m )
{
    _dim = m._dim;
    if ( _dim > 0 )
    {
        _pointValue = new ElementType[ _dim ];
        size_t i;
        for ( i = 0; i < _dim; ++i ) _pointValue[i] = m._pointValue[i];
    }
    else _pointValue = 0;
    return *this;
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
    ElementType* tmp;
    if ( dim > 0 )
    {
        tmp = new ElementType[ dim ];
        size_t i;
        for ( i = 0; i < dim; i++ ) tmp[i] = v[i];
    }
    else tmp = 0; 
    v = _pointValue;
    _pointValue = tmp;
    _dim = dim;
    if ( v ) delete[] v;
}


template< typename ElementType >
const ElementType&
m4dPoint< ElementType >::operator[]( size_t i ) const
{
    if ( i < _dim ) return _pointValue[i];
    else throw ErrorHandling::ExceptionBase( "Index out of bounds." );
}

    
template< typename ElementType >
ElementType&
m4dPoint< ElementType >::operator[]( size_t i )
{
    if ( i < _dim ) return _pointValue[i];
    else throw ErrorHandling::ExceptionBase( "Index out of bounds." );
}

template< typename ElementType >
const ElementType&
m4dPoint< ElementType >::getParticularValue( size_t i ) const
{
    if ( i < _dim ) return _pointValue[i];
    else throw ErrorHandling::ExceptionBase( "Index out of bounds." );
}

    
template< typename ElementType >
ElementType&
m4dPoint< ElementType >::getParticularValue( size_t i )
{
    if ( i < _dim ) return _pointValue[i];
    else throw ErrorHandling::ExceptionBase( "Index out of bounds." );
}

template< typename ElementType >
float
m4dPoint< ElementType >::distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 )
{
    if ( p1.getDimension() != p2.getDimension() ) throw ErrorHandling::ExceptionBase( "Dimension mismatch." );

    size_t i;
    float dist = 0;
    for ( i = 0; i < p1.getDimension(); ++i ) dist += ( p1.getParticularValue( i ) - p2.getParticularValue( i ) ) *
    						      ( p1.getParticularValue( i ) - p2.getParticularValue( i ) );
    return sqrt( dist );
}

template< typename ElementType >
m4dPoint< ElementType >
m4dPoint< ElementType >::midpoint( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 )
{
    if ( p1.getDimension() != p2.getDimension() ) throw ErrorHandling::ExceptionBase( "Dimension mismatch." );

    ElementType* tmp = new ElementType[ p1.getDimension() ];
    size_t i;
    for ( i = 0; i < p1.getDimension(); ++i ) tmp[i] = ( p1[i] + p2[i] ) / 2;
    m4dPoint< ElementType > p;
    p.setAllValues( tmp, p1.getDimension() );
    delete[] tmp;
    return p;
}

template< typename ElementType >
m4dShape< ElementType >::m4dShape( size_t dim, SliceOrientation so )
{
    _sliceOrientation = so;
    _dim      = dim;
    openShape();
}

template< typename ElementType >
void
m4dShape< ElementType >::setOrientation( SliceOrientation so )
{
    _sliceOrientation = so;
    calculateCentroid();
}

template< typename ElementType >
void
m4dShape< ElementType >::addPoint( m4dPoint< ElementType >& p )
{
    if ( p.getDimension() != _dim ) throw ErrorHandling::ExceptionBase( "Dimension mismatch." );
    else
    {
        if ( _shapePoints.size() > 1 ) _segmentLengths.push_back( m4dPoint< ElementType >::distance( _shapePoints.back(), p ) );
        _shapePoints.push_back( p );
    }
    calculateCentroid();
}

template< typename ElementType >
void
m4dShape< ElementType >::addAllPoints( m4dShape< ElementType >& s )
{
    if ( s.getDimension() != _dim ) throw ErrorHandling::ExceptionBase( "Index out of bounds." );
    else for ( typename std::list< m4dPoint< ElementType > >::iterator it = s._shapePoints.begin(); it != s._shapePoints.end(); ++it )
        addPoint( *it );
    calculateCentroid();
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
const std::list< float >&
m4dShape< ElementType >::segmentLengths() const
{
    return _segmentLengths;
}

template< typename ElementType >
std::list< float >&
m4dShape< ElementType >::segmentLengths()
{
    return _segmentLengths;
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
    if ( ! _shapePoints.empty() )    _shapePoints.pop_back();
    if ( ! _segmentLengths.empty() ) _segmentLengths.pop_back();
    calculateCentroid();
}

template< typename ElementType >
void
m4dShape< ElementType >::closeShape()
{
    _closed = true;
    calculateCentroid();
}

template< typename ElementType >
void
m4dShape< ElementType >::openShape()
{
    _closed = false;
    calculateCentroid();
}

template< typename ElementType >
bool
m4dShape< ElementType >::shapeClosed()
{
    return _closed;
}

template< typename ElementType >
size_t
m4dShape< ElementType >::getDimension()
{
    return _dim;
}

template< typename ElementType >
const m4dPoint< ElementType >&
m4dShape< ElementType >::getCentroid() const
{
    return _centroid;
}

template< typename ElementType >
long double
m4dShape< ElementType >::getArea() const
{
    return _area;
}

template< typename ElementType >
void
m4dShape< ElementType >::calculateCentroid()
{
    _centroid = m4dPoint< ElementType >();
    calculateArea();
    if ( !_closed || _shapePoints.size() < 3 || !_area ) return;
    ElementType coords[3];
    coords[ _sliceOrientation             ] = 0;
    coords[ ( _sliceOrientation + 1 ) % 3 ] = 0;
    coords[ ( _sliceOrientation + 2 ) % 3 ] = _shapePoints.front().getParticularValue( ( _sliceOrientation + 2 ) % 3 );
    m4dPoint< ElementType > p2, p1 = _shapePoints.back();
    for ( typename std::list< m4dPoint< ElementType > >::iterator it = _shapePoints.begin(); it != _shapePoints.end(); ++it )
    {
        p2 = *it;
	coords[ _sliceOrientation             ] += (ElementType)(( p1.getParticularValue( _sliceOrientation ) + p2.getParticularValue( _sliceOrientation ) ) * ( p1.getParticularValue( _sliceOrientation ) * p2.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) - p2.getParticularValue( _sliceOrientation ) * p1.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) ) / ( 6. * _area ));
	coords[ ( _sliceOrientation + 1 ) % 3 ] += (ElementType)(( p1.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) + p2.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) ) * ( p1.getParticularValue( _sliceOrientation ) * p2.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) - p2.getParticularValue( _sliceOrientation ) * p1.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) ) / ( 6. * _area ));
	p1 = p2;
    }
    coords[ _sliceOrientation             ] = ( coords[ _sliceOrientation            ]<0 ? -coords[ _sliceOrientation             ] : coords[ _sliceOrientation             ] );
    coords[ ( _sliceOrientation + 1 ) % 3 ] = ( coords[ ( _sliceOrientation + 1 ) % 3]<0 ? -coords[ ( _sliceOrientation + 1 ) % 3 ] : coords[ ( _sliceOrientation + 1 ) % 3 ] );
    _centroid = m4dPoint< ElementType >( coords[0], coords[1], coords[2] );
}

template< typename ElementType >
void
m4dShape< ElementType >::calculateArea()
{
    _area = 0.;
    if ( !_closed || _shapePoints.size() < 3 || _shapePoints.front().getDimension() < 3 ) return;
    m4dPoint< ElementType > p2, p1 = _shapePoints.back();
    for ( typename std::list< m4dPoint< ElementType > >::iterator it = _shapePoints.begin(); it != _shapePoints.end(); ++it )
    {
        p2 = *it;
	if ( p2.getParticularValue( ( _sliceOrientation + 2 ) % 3 ) != p1.getParticularValue( ( _sliceOrientation + 2 ) % 3 ) )
	{
	    _area = 0.;
	    return;
	}
	_area += p1.getParticularValue( _sliceOrientation ) * p2.getParticularValue( ( _sliceOrientation + 1 ) % 3 );
	_area -= p1.getParticularValue( ( _sliceOrientation + 1 ) % 3 ) * p2.getParticularValue( _sliceOrientation );
	p1 = p2;
    }
    _area /= 2.;
    _area = ( _area < 0 ? -_area : _area );
}

}/* namespace Selection */
}/* namespace M4D	*/

#endif /* m4dSelection.tcc */

/** @} */

