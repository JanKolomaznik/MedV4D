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
    
    m4dPoint();
    
    m4dPoint( ElementType x );
    
    m4dPoint( ElementType x, ElementType y );

    m4dPoint( ElementType x, ElementType y, ElementType z );
    
    m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t );

    m4dPoint( const m4dPoint< ElementType >& m );

    ~m4dPoint();

    m4dPoint&
    operator=( const m4dPoint< ElementType >& m );


    size_t
    getDimension();
    

    const ElementType*
    getAllValues() const;
    
    void
    setAllValues( ElementType* v, size_t dim );


    const ElementType&
    getParticularValue( size_t i ) const;
    
    ElementType&
    getParticularValue( size_t i );

    static float
    distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

private:
    
    size_t _dim;
    
    ElementType* _pointValue;

};

template< typename ElementType >
class m4dShape
{

public:

    m4dShape( size_t dim );

    void
    addPoint( m4dPoint< ElementType >& p );

    void
    addAllPoints( m4dShape< ElementType >& s);

    const std::list< m4dPoint< ElementType > >&
    shapeElements() const;

    std::list< m4dPoint< ElementType > >&
    shapeElements();

    const std::list< float >&
    segmentLengths() const;

    std::list< float >&
    segmentLengths();

    void
    clear();

    void
    deleteLast();

    void
    closeShape();

    void
    openShape();

    bool
    shapeClosed();

    size_t
    getDimension();

    const m4dPoint< ElementType >&
    getCentroid() const;

    float
    getArea() const;

private:

    std::list< m4dPoint< ElementType > >	_shapePoints;
    std::list< float >				_segmentLengths;
    m4dPoint< ElementType >			_centroid;
    float					_area;
    bool					_closed;
    size_t					_dim;

};

}/* namespace Selection */
}/* namespace M4D	*/

#include "m4dSelection.tcc" /* template sources */

#endif
