#ifndef _m4dSelection_h
#define _m4dSelection_h

#include <list>

namespace M4D
{

namespace Selection
{

/** m4dPoint
 *  class to represent a point in a 1-, 2-, 3- or 4-dimensional euclidian space
 */
template< typename ElementType >
class m4dPoint
{

public:
    
    /** constructor
     *  to create an empty point
     */
    m4dPoint();
    
    /** constructor
     *  to create a 1-dimensional point
     *   @param x the first dimension coordinate of the point
     */
    m4dPoint( ElementType x );
    
    /** constructor
     *  to create a 2-dimensional point
     *   @param x the first dimension coordinate of the point
     *   @param y the second dimension coordinate of the point
     */
    m4dPoint( ElementType x, ElementType y );

    /** constructor
     *  to create a 3-dimensional point
     *   @param x the first dimension coordinate of the point
     *   @param y the second dimension coordinate of the point
     *   @param z the third dimension coordinate of the point
     */
    m4dPoint( ElementType x, ElementType y, ElementType z );
    
    /** constructor
     *  to create 4-dimensional point
     *   @param x the first dimension coordinate of the point
     *   @param y the second dimension coordinate of the point
     *   @param z the third dimension coordinate of the point
     *   @param t the fourth dimension coordinate of the point
     */
    m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t );

    /** constructor
     *  to create a point by copying another one
     *   @param m reference to the point to be copyied
     */
    m4dPoint( const m4dPoint< ElementType >& m );

    /** destructor
     *  to destroy the point object
     */
    ~m4dPoint();

    /** operator=
     *  to copy another point
     *   @param m reference to the point to be copyied
     *   @return reference to the newly copied point
     */
    m4dPoint&
    operator=( const m4dPoint< ElementType >& m );


    /** getDimension()
     *   @return the number of dimensions of the point's euclidean space
     */
    size_t
    getDimension();
    

    /** getAllValues()
     *   @return pointer to the array of coordinates of the point
     */
    const ElementType*
    getAllValues() const;
    
    /** setAllValues()
     *  Sets all the coordinates of the point
     *   @param v pointer to the array of new coordinates to be set
     *   @param dim the number of dimensions of the point's euclidean space
     */
    void
    setAllValues( ElementType* v, size_t dim );


    /** operator[]
     *   @param i the number of the dimension that's value is to be returned
     *   @return a constant reference to i-th dimension value
     */
    const ElementType&
    operator[]( size_t i ) const;
    
    /** operator[]
     *   @param i the number of the dimension that's value is to be returned
     *   @return a reference to i-th dimension value
     */
    ElementType&
    operator[]( size_t i );
 
    /** getParticularValue()
     *   @param i the number of the dimension that's value is to be returned
     *   @return a constant reference to i-th dimension value
     */
    const ElementType&
    getParticularValue( size_t i ) const;
    
    /** getParticularValue()
     *   @param i the number of the dimension that's value is to be returned
     *   @return a reference to i-th dimension value
     */
    ElementType&
    getParticularValue( size_t i );

    /** distance
     *  static function to calculate the distance between two points
     *   @param p1 the first point
     *   @param p2 the second point
     *   @return the distance between the two points
     */
    static float
    distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

    /** midpoint
     *  static function to calculate the midpoint between two points
     *   @param p1 the first point
     *   @param p2 the second point
     *   @return a point that lies in the middle of segment |p1 p2|
     */
    static m4dPoint< ElementType >
    midpoint( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

private:
    
    /** _dim
     *  the number of dimensions
     */
    size_t _dim;
    
    /** _pointValue
     *  the coordinate values
     */
    ElementType* _pointValue;

};

/** m4dShape
 *  class to represent a shape that is built up from m4dPoints
 */
template< typename ElementType >
class m4dShape
{

public:

    /** constructor
     *  to create a new empty shape
     *   @param dim the dimension of the points in the shape
     */
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

    long double
    getArea() const;

    void
    calculateCentroid();

    void
    calculateArea();

private:

    std::list< m4dPoint< ElementType > >	_shapePoints;
    std::list< float >				_segmentLengths;
    m4dPoint< ElementType >			_centroid;
    long double					_area;
    bool					_closed;
    size_t					_dim;

};

}/* namespace Selection */
}/* namespace M4D	*/

#include "m4dSelection.tcc" /* template sources */

#endif
