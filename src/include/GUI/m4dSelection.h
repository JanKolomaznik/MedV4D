/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file m4dSelection.h 
 * @{ 
 **/

#ifndef _m4dSelection_h
#define _m4dSelection_h

#include <list>

namespace M4D
{
    
    /**
     * Enumeration that tells according to which axis the 3D image's slices are viewed
     */
    typedef enum { xy = 0, yz = 1, zx = 2 } SliceOrientation;


namespace Selection
{

/**
 *  class to represent a point in a 1-, 2-, 3- or 4-dimensional euclidian space
 */
template< typename ElementType >
class m4dPoint
{

public:
    
    /**
     *  constructor to create an empty point
     */
    m4dPoint();
    
    /**
     *  constructor to create a 1-dimensional point
     *   @param x the first dimension coordinate of the point
     */
    m4dPoint( ElementType x );
    
    /**
     *  constructor to create a 2-dimensional point
     *   @param x the first dimension coordinate of the point
     *   @param y the second dimension coordinate of the point
     */
    m4dPoint( ElementType x, ElementType y );

    /**
     *  constructor to create a 3-dimensional point
     *   @param x the first dimension coordinate of the point
     *   @param y the second dimension coordinate of the point
     *   @param z the third dimension coordinate of the point
     */
    m4dPoint( ElementType x, ElementType y, ElementType z );
    
    /**
     *  constructor to create 4-dimensional point
     *   @param x the first dimension coordinate of the point
     *   @param y the second dimension coordinate of the point
     *   @param z the third dimension coordinate of the point
     *   @param t the fourth dimension coordinate of the point
     */
    m4dPoint( ElementType x, ElementType y, ElementType z, ElementType t );

    /**
     *  constructor to create a point by copying another one
     *   @param m reference to the point to be copyied
     */
    m4dPoint( const m4dPoint< ElementType >& m );

    /**
     *  to destroy the point object
     */
    ~m4dPoint();

    /**
     *  constructor to copy another point
     *   @param m reference to the point to be copyied
     *   @return reference to the newly copied point
     */
    m4dPoint&
    operator=( const m4dPoint< ElementType >& m );


    /**
     *   @return the number of dimensions of the point's euclidean space
     */
    size_t
    getDimension();
    

    /**
     *   @return pointer to the array of coordinates of the point
     */
    const ElementType*
    getAllValues() const;
    
    /**
     *  Sets all the coordinates of the point
     *   @param v pointer to the array of new coordinates to be set
     *   @param dim the number of dimensions of the point's euclidean space
     */
    void
    setAllValues( ElementType* v, size_t dim );


    /**
     *   @param i the number of the dimension that's value is to be returned
     *   @return a constant reference to i-th dimension value
     */
    const ElementType&
    operator[]( size_t i ) const;
    
    /**
     *   @param i the number of the dimension that's value is to be returned
     *   @return a reference to i-th dimension value
     */
    ElementType&
    operator[]( size_t i );
 
    /**
     *   @param i the number of the dimension that's value is to be returned
     *   @return a constant reference to i-th dimension value
     */
    const ElementType&
    getParticularValue( size_t i ) const;
    
    /**
     *   @param i the number of the dimension that's value is to be returned
     *   @return a reference to i-th dimension value
     */
    ElementType&
    getParticularValue( size_t i );

    /**
     *  static function to calculate the distance between two points
     *   @param p1 the first point
     *   @param p2 the second point
     *   @return the distance between the two points
     */
    static float
    distance( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

    /**
     *  static function to calculate the midpoint between two points
     *   @param p1 the first point
     *   @param p2 the second point
     *   @return a point that lies in the middle of segment |p1 p2|
     */
    static m4dPoint< ElementType >
    midpoint( m4dPoint< ElementType >& p1, m4dPoint< ElementType >& p2 );

private:
    
    /**
     *  the number of dimensions
     */
    size_t _dim;
    
    /**
     *  the coordinate values
     */
    ElementType* _pointValue;

};

/**
 *  class to represent a shape that is built up from m4dPoints
 */
template< typename ElementType >
class m4dShape
{

public:

    /**
     *  constructor to create a new empty shape
     *   @param dim the dimension of the points in the shape
     *   @param so the slice orientation of the viewer
     */
    m4dShape( size_t dim, SliceOrientation so );

    /**
     *  set the orientation of the viewer
     *   @param so the slice orientation of the viewer
     */
    void
    setOrientation( SliceOrientation so );

    /**
     *  adds a new point to the shape
     *   @param p reference to the new point
     */
    void
    addPoint( m4dPoint< ElementType >& p );

    /**
     *  adds all points of another shape to this one
     *   @param s reference to the other shape
     */
    void
    addAllPoints( m4dShape< ElementType >& s);

    /**
     *  get a constant list of all the points of the shape
     *   @return const list of the points
     */
    const std::list< m4dPoint< ElementType > >&
    shapeElements() const;

    /**
     *  get a list of all the points of the shape
     *   @return list of the points
     */
    std::list< m4dPoint< ElementType > >&
    shapeElements();

    /**
     *  get a constant list of the lengths of the segments of the shape
     *   @return const list of lengths
     */
    const std::list< float >&
    segmentLengths() const;

    /**
     *  get a list of the lengths of the segments of the shape
     *   @return list of lengths
     */
    std::list< float >&
    segmentLengths();

    /**
     *  clear the shape - erase all its points
     */
    void
    clear();

    /**
     *  deletes the point that was last added to the shape
     */
    void
    deleteLast();

    /**
     *  closes the shape ( the first and last points will be the same )
     */
    void
    closeShape();

    /**
     *  opens the shape ( the first and last points will be different )
     */
    void
    openShape();

    /**
     *  checks if the shape is closed
     *   @return true if the shape is closed, false otherwise
     */
    bool
    shapeClosed();

    /**
     *  get the number of dimensions of the euclidean space that the shape lies in
     *   @return the dimension number
     */
    size_t
    getDimension();

    /**
     *  get the centroid of the shape
     *   @return the centroid point
     */
    const m4dPoint< ElementType >&
    getCentroid() const;

    /**
     *  get the area size of the shape
     *   @return the size of the area
     */
    long double
    getArea() const;

private:
    
    /**
     *  (re-)calculate the centroid of the shape
     */
    void
    calculateCentroid();

    /**
     *  (re-)calculate the area size of the shape
     */
    void
    calculateArea();


    /**
     *  list of points of the shape
     */
    std::list< m4dPoint< ElementType > >	_shapePoints;
    
    /**
     *  list of segment lengths of the shape
     */
    std::list< float >				_segmentLengths;
    
    /**
     *  centroid point of the shape ( undefined if shape is open )
     */
    m4dPoint< ElementType >			_centroid;
    
    /**
     *  area size of the shape ( 0 if shape is open )
     */
    long double					_area;

    /**
     *  true if shape is closed, false otherwise
     */
    bool					_closed;
    
    /**
     *  the number of dimensions of the points of the shape
     */
    size_t					_dim;

    /**
     *  the orientation of the viewer
     */
    SliceOrientation				_sliceOrientation;

};

}/* namespace Selection */
}/* namespace M4D	*/

#include "m4dSelection.tcc" /* template sources */

#endif

/** @} */

