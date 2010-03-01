#ifndef POINT_SET_H
#define POINT_SET_H

#include "common/Common.h"
#include "Imaging/GeometricalObject.h"
#include "Imaging/GeometricAlgorithms.h"
#include "common/Vector.h"
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

	
template < typename VectorType >
class PointSet: public AGeometricalObjectDimPrec< VectorType >
{
public:
	typedef AGeometricalObjectDimPrec< VectorType >		PredecessorType;
	typedef typename PredecessorType::PointType 		PointType;
	typedef typename PredecessorType::Type			Type;
	typedef std::vector< PointType >			PointVector;
	typedef PointSet< VectorType >				ThisType;
	typedef typename PointVector::iterator			Iterator;
	typedef typename PointVector::const_iterator		ConstIterator;
	friend void SerializeGeometryObject< VectorType >( M4D::IO::OutStream &stream, const ThisType &obj );
	//friend void DeserializeGeometryObject< VectorType >( M4D::IO::InStream &stream, ThisType * &obj ); 
	static const unsigned Dimension	= VectorType::Dimension;		

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
	Scale( Vector< float32, Dimension > factors, PointType center )
		{
			std::for_each( _points.begin(), _points.end(), ScaleFunctor< PointType >( factors, center ) );
		}

	Iterator
	Begin()	{ return _points.begin(); }

	Iterator
	End()	{ return _points.end(); }

	ConstIterator
	Begin() const { return _points.begin(); }

	ConstIterator
	End() const { return _points.end(); }

protected:
	PointVector	_points;
	uint32		_pointCount;
};

template < typename VectorType >
void
PrintPointSet( std::ostream &stream, const PointSet< VectorType > &set, unsigned step = 1 )
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
	typename PointSetType::Type distanceSq = (pset[0]-point)*(pset[0]-point);
	int32 idx = 0;
	for( unsigned i = 1; i < pset.Size(); ++i ) {
		typename PointSetType::Type tmp = (pset[i]-point)*(pset[i]-point);
		if( tmp < distanceSq ) {
			distanceSq = tmp;
			idx = i;
		}
	}
	return idx;
}

template< typename VectorType >
void 
SerializeGeometryObject( M4D::IO::OutStream &stream, const PointSet< VectorType > &obj )
{
		stream.Put<uint32>( GMN_BEGIN_ATRIBUTES );
			stream.Put( DummySpace< 5 >() );
			stream.Put<uint32>( obj._pointCount );
		stream.Put<uint32>( GMN_END_ATRIBUTES );

		stream.Put<uint32>( GMN_BEGIN_DATA );
			for( uint32 i = 0; i < obj._pointCount; ++i ) {
				stream.Put< VectorType >( obj._points[i] );
			}
		stream.Put<uint32>( GMN_END_DATA );
}

template< typename VectorType >
void
DeserializeGeometryObject( M4D::IO::InStream &stream, PointSet< VectorType > * &obj )
{
		_THROW_ M4D::ErrorHandling::ETODO();

}

}/*namespace Geometry*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*POINT_SET_H*/
