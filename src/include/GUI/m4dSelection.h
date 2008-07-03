#ifndef _m4dSelection_h
#define _m4dSelection_h

#include <list>

namespace M4D
{

namespace Selection
{

template< typename ElementType >
class m4dPoint
{

public:
    
    m4dPoint( ElementType x );
    
    m4dPoint( ElementType x, ElementType y );

    m4dPoint( ElementType x, ElementType y, ElementType z );
    
    m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t );


    size_t
    GetDimension();
    

    const ElementType*
    GetAllValues() const;
    
    void
    SetAllValues( ElementType* v, size_t dim );


    const ElementType&
    GetParticularValue( size_t i ) const;
    
    ElementType&
    GetParticularValue( size_t i );

    static float
    Distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

private:
    
    size_t _dim;
    
    ElementType* _pointValue;

};

template< typename ElementType >
class m4dSegment
{

public:
    
    m4dLine( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

    m4dPoint< ElementType >&
    GetSegmentStartPoint();

    m4dPoint< ElementType >&
    GetSegmentStartPoint() const;

    m4dPoint< ElementType >&
    GetSegmentEndPoint();

    m4dPoint< ElementType >&
    GetSegmentEndPoint() const;

    bool
    PointLiesInsideSegment( m4dPoint< ElementType >& p );

private:

    m4dPoint< ElementType > _startPoint;
    m4dPoint< ElementType > _endPoint;
    
};

template< typename ElementType >
class m4dShape
{

public:

    m4dShape();

    void
    addPoint( m4dPoint< ElementType >& p );

    void
    addAllPoints( m4dShape< ElementType >& s);

    const list< m4dPoint< ElementType > >&
    ShapeElements() const;

    list< m4Point< ElementType > >&
    ShapeElements();

    void
    closeShape();

    void
    openShape();

    bool
    shapeClosed();

private:

    list< m4dPoint< ElementType > >	_shapePoints;
    bool				_closed;

};

}/* namespace Selection */
}/* namespace M4D	*/

#include "m4dSelection.tcc" /* template sources */

#endif
