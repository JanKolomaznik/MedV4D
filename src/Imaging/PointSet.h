#ifndef POINT_SET_H
#define POINT_SET_H

#include "Common.h"
#include "Imaging/GeometricalObject.h"
#include "Coordinates.h"
#include <vector>
#include <ostream>
#include <iomanip>


namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file PointSet.h 
 * @{ 
 **/

namespace Imaging
{
namespace Geometry
{

	
template < typename CoordType, unsigned Dim >
class PointSet: public AGeometricalObjectDimPrec< CoordType, Dim >
{
public:
	typedef AGeometricalObjectDimPrec< CoordType, Dim >	PredecessorType;
	typedef typename PredecessorType::PointType 		PointType;
	typedef typename PredecessorType::Type			Type;
	typedef std::vector< PointType >			PointVector;
	static const unsigned Dimension	= Dim;		

	PointSet() : _points(), _pointCount( 0 ) 
		{}

	void
	Resize( unsigned size )
		{ _points.resize( _pointCount = size ); }

	uint32
	Size()const
		{ return _pointCount; }

	PointType &
	operator[]( unsigned idx ) 
		{ 
			if( idx < _pointCount ) {
				return _points[ idx ]; 
			} else 
				throw ErrorHandling::EBadIndex(); 
		}

	const PointType &
	operator[]( unsigned idx ) const
		{ 
			if( idx < _pointCount ) {
				return _points[ idx ]; 
			} else 
				throw ErrorHandling::EBadIndex(); 
		}

	typename PointVector::iterator
	Begin()
		{ return _points.begin(); }
	
	typename PointVector::const_iterator
	Begin()const
		{ return _points.begin(); }

	typename PointVector::iterator
	End()
		{ return _points.end(); }
	
	typename PointVector::const_iterator
	End()const
		{ return _points.end(); }

	PointType &
	GetPointCyclic( unsigned idx ) 
		{ 
			return _points[ MOD( idx, _pointCount ) ]; 
		}

	const PointType &
	GetPointCyclic( unsigned idx ) const
		{ 
			return _points[ MOD( idx, _pointCount ) ]; 
		}

	PointType &
	GetPointACyclic( unsigned idx ) 
		{ 
			return _points[ Max( Min( idx, _pointCount-1 ), 0 ) ]; 
		}

	const PointType &
	GetPointACyclic( unsigned idx ) const
		{ 
			return _points[ Max( Min( idx, _pointCount-1 ), 0 ) ]; 
		}

	size_t
	AddPoint( const PointType &point )
		{ _points.push_back( point ); return _pointCount++; } 

	void
	InsertPoint( unsigned before, const PointType &point )
		{ _points.insert( _points.begin() + before, point ); ++_pointCount; } 

	void
	RemovePoint( unsigned idx )
		{
			_points.erase( _points.begin() + idx );	
			_pointCount = _points.size();
		}
	/**
	 * Remove points from interval [first,last).
	 * If first > last -> removing cyclically.
	 **/
	void
	RemovePoints( unsigned first, unsigned last )
		{
			if( first < last ) {
				_points.erase( _points.begin() + first, _points.begin() + last );
			} else {
				_points.erase( _points.begin() + first, _points.end() );
				_points.erase( _points.begin(), _points.begin() + last );
			}
			_pointCount = _points.size();
		}
protected:
	PointVector	_points;
	uint32		_pointCount;
};

template < typename CoordType, unsigned Dim >
void
PrintPointSet( std::ostream &stream, const PointSet< CoordType, Dim > &set )
{
	for( size_t i = 0; i < set.Size(); ++i ) {
		stream << set[i] << std::endl;
	}
}

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POINT_SET_H*/
