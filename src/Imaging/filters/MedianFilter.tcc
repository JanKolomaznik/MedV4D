#ifndef _MEDIAN_FILTER_H
#error File MedianFilter.tcc cannot be included directly!
#else

#include <algorithm>

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MedianFilter.tcc 
 * @{ 
 **/

namespace Imaging
{

template< typename ElType >
class MedianFilterFtor: public FilterFunctorBase< ElType >
{
public:
	MedianFilterFtor( uint32 radius ) : _radius( radius ), _lastRow( TypeTraits< int32 >::Min ), _size( (2*radius+1) )
	{ _array = new ElType[ _size*_size ]; }

	MedianFilterFtor( const MedianFilterFtor &ftor ): _radius( ftor._radius ), _lastRow( TypeTraits< int32 >::Min ), _size( (2*ftor._radius+1) )
	{ _array = new ElType[ _size*_size ]; }

	~MedianFilterFtor()
	{ delete [] _array; }

	template< typename Accessor >
	ElType
	Apply( const Vector< int32, 2 > &pos, Accessor &accessor ) 
	{
		Vector< int32, 2 > idx;
		unsigned i = 0;
/*		if( _lastRow == pos[1] ) {
			idx[0] = pos[0] + _radius;
			for( idx[1] = pos[1] - _radius; idx[1] <= pos[1] + _radius; ++idx[1] ) {
				_array[ i*_size + _lastCol ] = accessor( idx );
				++i;
			}
			_lastCol = (_lastCol + 1) % _size;
		} else {*/
			for( idx[1] = pos[1] - _radius; idx[1] <= pos[1] + _radius; ++idx[1] ) {
				for( idx[0] = pos[0] - _radius; idx[0] <= pos[0] + _radius; ++idx[0] ) {
					_array[i++] = accessor( idx );
				}
			}
			_lastCol = 0;
			_lastRow = pos[1];
		//}
		std::nth_element( &(_array[ 0 ]), &(_array[ _size*_size / 2 ]), &(_array[ _size*_size ]) );
		return _array[ _size*_size / 2 ];
	}
	Vector< int32, 2 >
	GetLeftCorner()const
	{ return Vector< int32, 2 >( -_radius, -_radius ); }

	Vector< int32, 2 >
	GetRightCorner()const
	{ return Vector< int32, 2 >( _radius, _radius ); }
protected:
	int32	_radius;
	int32	_lastRow;
	int32	_lastCol;
	uint32	_size;
	ElType	*_array;
};


template< typename InputImageType >
MedianFilter2D< InputImageType >
::MedianFilter2D() : PredecessorType( new Properties() )
{

}

template< typename InputImageType >
MedianFilter2D< InputImageType >
::MedianFilter2D( typename MedianFilter2D< InputImageType >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename InputImageType >
void
MedianFilter2D< InputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	if( utype != AbstractPipeFilter::RECALCULATION 
		&& this->_propertiesTimestamp != GetProperties().GetTimestamp() )
	{
		utype = AbstractPipeFilter::RECALCULATION;
	}
}

template< typename InputImageType >
bool
MedianFilter2D< InputImageType >
::Process2D(
			const typename MedianFilter2D< InputImageType >::Region	&inRegion,
			typename MedianFilter2D< InputImageType >::Region 	&outRegion
		 )
{
	if( !this->CanContinue() ) {
		return false;
	}

	try {
		MedianFilterFtor< InputElementType > filter( GetRadius() );
		FilterProcessorNeighborhood< 
			MedianFilterFtor< InputElementType >,
			Region,
			Region,
			MirrorAccessor
			>( filter, inRegion, outRegion );
	}
	catch( ... ) { 
		return false; 
	}
	
	return true;
}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/


#endif /*_MEDIAN_FILTER_H*/


