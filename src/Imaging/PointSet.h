#ifndef POINT_SET_H
#define POINT_SET_H

#include "Common.h"
#include "Imaging/GeometricalObject.h"
#include "Imaging/GeometricAlgorithms.h"
#include "Vector.h"
#include <vector>
#include <ostream>
#include <iomanip>
#include <algorithm>


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
	typedef CoordType					CoordinateType;
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
				_THROW_ ErrorHandling::EBadIndex( TO_STRING("Index = " << idx << " out of range < 0, " << Size() << " ).") ); 
		}

	const PointType &
	operator[]( unsigned idx ) const
		{ 
			if( idx < _pointCount ) {
				return _points[ idx ]; 
			} else 
				_THROW_ ErrorHandling::EBadIndex( TO_STRING("Index = " << idx << " out of range < 0, " << Size() << " ).") ); 
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
	GetPointCyclic( int32 idx ) 
		{ 
			return _points[ MOD( idx, _pointCount ) ]; 
		}

	const PointType &
	GetPointCyclic( int32 idx ) const
		{ 
			return _points[ MOD( idx, _pointCount ) ]; 
		}

	PointType &
	GetPointACyclic( int32 idx ) 
		{ 
			return _points[ Max( Min( idx, (int32)_pointCount-1 ), 0 ) ]; 
		}

	const PointType &
	GetPointACyclic( int32 idx ) const
		{ 
			return _points[ Max( Min( idx, (int32)_pointCount-1 ), 0 ) ]; 
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

	void
	Move( PointType t )
		{
			std::for_each( _points.begin(), _points.end(), MoveFunctor< PointType >( t ) );
		}

	void
	Scale( Vector< float32, Dim > factors, PointType center )
		{
			std::for_each( _points.begin(), _points.end(), ScaleFunctor< PointType >( factors, center ) );
		}
	
protected:
	PointVector	_points;
	uint32		_pointCount;
};

template < typename CoordType, unsigned Dim >
void
PrintPointSet( std::ostream &stream, const PointSet< CoordType, Dim > &set, unsigned step = 1 )
{
	if( step == 0 ) {
		step = 1;
	}
	for( size_t i = 0; i < set.Size(); i+=step ) {
		stream << set[i] << std::endl;
	}
}

template< typename PointSetType >
int32
ClosestPointFromPointSet( const PointSetType &pset, typename PointSetType::PointType point )
{
	if( 0 == pset.Size() ) {
		return -1;
	}
	typename PointSetType::CoordinateType distanceSq = (pset[0]-point)*(pset[0]-point);
	int32 idx = 0;
	for( unsigned i = 1; i < pset.Size(); ++i ) {
		typename PointSetType::CoordinateType tmp = (pset[i]-point)*(pset[i]-point);
		if( tmp > distanceSq ) {
			distanceSq = tmp;
			idx = i;
		}
	}
	return idx;
}

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POINT_SET_H*/
