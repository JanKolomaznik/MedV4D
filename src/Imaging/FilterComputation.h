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
#include "Imaging/ImageRegionAccessors.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename ElType >
class FilterFunctorBase
{
public:
	typedef ElType OutputValue;
};

template< typename In, typename Out >
class PreprocessorBase
{
public:
	typedef In	InputValue;
	typedef Out 	OutputValue;
};

template< typename Filter, typename Accessor >
class BasicFilterApplicator
{
public:
	//typedef Vector< int32, 2 > CoordType;
	typedef typename Accessor::CoordType CoordType;

	BasicFilterApplicator( Filter &filter, Accessor &accessor ) : _filter( filter ), _accessor( accessor )
	{ }

	void
	operator()( typename Filter::OutputValue &value, const CoordType &coordinates ) {
		value = _filter.Apply( coordinates, _accessor );
	}

	Filter		&_filter; 
	Accessor	&_accessor;
};

template< typename Filter, typename Accessor, typename Preprocessor >
class PreprocessorFilterApplicator
{
public:
	//typedef Vector< int32, 2 > CoordType;
	typedef typename Accessor::CoordType CoordType;

	PreprocessorFilterApplicator( Filter &filter, Accessor &accessor, Preprocessor &preprocessor ) : _filter( filter ), _accessor( accessor ), _preprocessor( preprocessor )
	{ }

	void
	operator()( typename Preprocessor::OutputValue &value, const CoordType &coordinates ) {
		_preprocessor( _filter.Apply( coordinates, _accessor ), value );
	}

	Filter		&_filter; 
	Accessor	&_accessor;
	Preprocessor	_preprocessor;
};

template< typename Filter, typename InputRegion, typename OutputRegion, template< typename Region > class Accessor  >
void
FilterProcessor( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	typedef Accessor< InputRegion > AccessorType;
	AccessorType accessor( input );

	ForEachInRegion( output, BasicFilterApplicator< Filter, AccessorType >( filter, accessor ) );

}

template< typename OutputRegion, typename Applicator  >
void
SolveBoundaryFiltering2D( OutputRegion &output, const Applicator &applicator, const typename OutputRegion::PointType &leftCorner, const typename OutputRegion::PointType &rightCorner )
{
	//static const unsigned Dim = OutputRegion::Dimension;
	typedef typename OutputRegion::PointType PointType;
	PointType minimum = output.GetMinimum();
	PointType maximum = output.GetMaximum();

	PointType b0 = PointType( leftCorner[0], minimum[1] );
	PointType b1 = PointType( maximum[0], leftCorner[1] );
	typename OutputRegion::Iterator iterator = output.GetIterator( b0, b1 );
	ForEachByIterator( iterator, applicator );
		
	b0 = PointType( rightCorner[0], leftCorner[1] );
	b1 = maximum;
	iterator = output.GetIterator( b0, b1 );
	ForEachByIterator( iterator, applicator );
	
	b0 = PointType( minimum[0], rightCorner[1] );
	b1 = PointType( rightCorner[0], maximum[1] );
	iterator = output.GetIterator( b0, b1 );
	ForEachByIterator( iterator, applicator );

	b0 = minimum;
	b1 = PointType( leftCorner[0], rightCorner[1] );
	iterator = output.GetIterator( b0, b1 );
	ForEachByIterator( iterator, applicator );
}

template< typename Filter, typename InputRegion, typename OutputRegion, template< typename Region > class Accessor  >
void
FilterProcessorNeighborhood( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	typedef Accessor< InputRegion > AccessorType;
	typedef SimpleAccessor< InputRegion > SimpleAccessorType;
	AccessorType accessor( input );
	SimpleAccessorType simpleAccessor( input );
	
	typename OutputRegion::PointType minimum = output.GetMinimum();
	typename OutputRegion::PointType maximum = output.GetMaximum();
	typename OutputRegion::PointType leftCorner = minimum - filter.GetLeftCorner();
	typename OutputRegion::PointType rightCorner = maximum - filter.GetRightCorner();

	if( OutputRegion::Dimension == 2 ) {	
		SolveBoundaryFiltering2D< OutputRegion, BasicFilterApplicator< Filter, AccessorType > >
			( output, BasicFilterApplicator< Filter, AccessorType >( filter, accessor ), leftCorner, rightCorner );
	} else {
		_THROW_ ErrorHandling::ETODO();
	}
	
	typename OutputRegion::Iterator iterator = output.GetIterator( leftCorner, rightCorner );

	ForEachByIterator( iterator, BasicFilterApplicator< Filter, SimpleAccessorType >( filter, simpleAccessor ) );

}

template< typename Filter, typename InputRegion, typename OutputRegion  >
void
FilterProcessorNeighborhoodSimple( Filter &filter, const InputRegion &input, OutputRegion &output )
{
	typedef SimpleAccessor< InputRegion > SimpleAccessorType;
	SimpleAccessorType simpleAccessor( input );
	
	typename OutputRegion::PointType minimum = output.GetMinimum();
	typename OutputRegion::PointType maximum = output.GetMaximum();
	typename OutputRegion::PointType leftCorner = minimum - filter.GetLeftCorner();
	typename OutputRegion::PointType rightCorner = maximum - filter.GetRightCorner();

	/*if( OutputRegion::Dimension == 2 ) {	
		SolveBoundaryFiltering2D< OutputRegion, BasicFilterApplicator< Filter, AccessorType > >
			( output, BasicFilterApplicator< Filter, AccessorType >( filter, accessor ), leftCorner, rightCorner );
	} else {
		_THROW_ ErrorHandling::ETODO();
	}*/

	//TODO - put zeroes to boundaries
	
	typename OutputRegion::Iterator iterator = output.GetIterator( leftCorner, rightCorner );

	ForEachByIterator( iterator, BasicFilterApplicator< Filter, SimpleAccessorType >( filter, simpleAccessor ) );

}


template< typename Filter, typename InputRegion, typename OutputRegion, template< typename Region > class Accessor, template< typename In, typename Out > class Preprocessor  >
void
FilterProcessorNeighborhoodPreproc( 
			Filter			&filter, 
			const InputRegion	&input, 
			OutputRegion		&output,
			Preprocessor< typename Filter::OutputValue, typename OutputRegion::ElementType > preprocessor 
			)
{
	typedef Accessor< InputRegion > AccessorType;
	typedef SimpleAccessor< InputRegion > SimpleAccessorType;
	typedef Preprocessor< typename Filter::OutputValue, typename OutputRegion::ElementType > PreprocessorType;
	AccessorType accessor( input );
	SimpleAccessorType simpleAccessor( input );
	
	typename OutputRegion::PointType minimum = output.GetMinimum();
	typename OutputRegion::PointType maximum = output.GetMaximum();
	typename OutputRegion::PointType leftCorner = minimum - filter.GetLeftCorner();
	typename OutputRegion::PointType rightCorner = maximum - filter.GetRightCorner();

	if( OutputRegion::Dimension == 2 ) {	
		SolveBoundaryFiltering2D< OutputRegion, PreprocessorFilterApplicator< Filter, AccessorType, PreprocessorType > >
			( output, PreprocessorFilterApplicator< Filter, AccessorType, PreprocessorType >( filter, accessor, preprocessor ), leftCorner, rightCorner );
	} else {
		_THROW_ ErrorHandling::ETODO();
	}
	
	typename OutputRegion::Iterator iterator = output.GetIterator( leftCorner, rightCorner );

	ForEachByIterator( iterator, PreprocessorFilterApplicator< Filter, SimpleAccessorType, PreprocessorType >( filter, simpleAccessor, preprocessor ) );

}

/*template< typename Filter, typename Region, template< typename Region > class Accessor  >
void
FilterProcessorInPlace( Filter &filter, OutputRegion &region )
{
	typedef Accessor< InputRegion > AccessorType;
	AccessorType accessor( region );

	ForEachInRegion( output, FilterApplicator< Filter, AccessorType >( filter, accessor ) );

}*/

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */
#endif /*FILTER_COMPUTATION_H*/

/** @} */

