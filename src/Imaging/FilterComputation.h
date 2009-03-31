/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Convolution.h 
 * @{ 
 **/

#ifndef FILTER_COMPUTATION_H
#define FILTER_COMPUTATION_H

#include "Imaging/Image.h"
#include "common/Vector.h"
#include "Imaging/ImageRegion.h"
#include <boost/shared_ptr.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename RegionType >
class AccessorBase
{
public:
	static const unsigned Dimension = RegionType::Dimension;
	typedef Vector< int32, Dimension > CoordType;

	AccessorBase( const RegionType & region ): _region( region ) 
	{}

	const RegionType		&_region;

};

template< typename RegionType >
class SimpleAccessor: public AccessorBase< RegionType >
{
public:
	typedef AccessorBase< RegionType > PredecessorType;

	MirrorAccessor( const RegionType & region ): PredecessorType( region )
	{}

	typename RegionType::ElementType
	operator()( const Vector< int32, Dimension > &pos )const
	{
		return _region.GetElement( pos );
	}

};

template< typename RegionType >
class MirrorAccessor: public AccessorBase< RegionType >
{
public:
	typedef AccessorBase< RegionType > PredecessorType;

	MirrorAccessor( const RegionType & region ): PredecessorType( region ), _minimum( region.GetMinimum() )
	{
		_maximumIn = _region.GetMaximum() - Vector< int32, Dimension >( 1 );
	}

	typename RegionType::ElementType
	operator()( const Vector< int32, Dimension > &pos )const
	{
		if( pos >= _minimum && pos <= _maximumIn ) {
				return _region.GetElement( pos );
		}
		Vector< int32, Dimension > tmp = pos - _minimum;
		VectorAbs( tmp );
		tmp += _minimum;

		tmp = _maximumIn - tmp;
		VectorAbs( tmp );
		tmp = _maximumIn - tmp;

		return _region.GetElement( tmp );
	}

	Vector< int32, Dimension >	_minimum;
	Vector< int32, Dimension >	_maximumIn;
};

template< typename Filter, typename Accessor >
class FilterApplicator
{
public:
	//typedef Vector< int32, 2 > CoordType;
	typedef typename Accessor::CoordType CoordType;

	FilterApplicator( Filter &filter, Accessor &accessor ) : _filter( filter ), _accessor( accessor )
	{ }

	void
	operator()( typename Filter::OutputValue &value, const CoordType &coordinates ) {
		value = _filter.Apply( coordinates, _accessor );
	}

	Filter		&_filter; 
	Accessor	&_accessor;
};

template< typename Filter, typename InputRegion, typename OutputRegion, template< typename Region > class Accessor  >
void
FilterProcessor( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	typedef Accessor< InputRegion > AccessorType;
	AccessorType accessor( input );

	ForEachInRegion( output, FilterApplicator< Filter, AccessorType >( filter, accessor ) );

}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */
#endif /*FILTER_COMPUTATION_H*/

/** @} */

