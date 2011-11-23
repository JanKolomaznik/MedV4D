#ifndef _MIN_FILTER_H
#error File MinFilter.tcc cannot be included directly!
#else

#include <algorithm>

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MinFilter.tcc 
 * @{ 
 **/

namespace Imaging
{

template< typename ElType >
class MinFilterFtor: public FilterFunctorBase< ElType >
{
public:
	MinFilterFtor( uint32 radius ) : _radius( radius ), _lastRow( TypeTraits< int32 >::Min ), _size( (2*radius+1) )
	{ _array = new ElType[ _size*_size ]; }

	MinFilterFtor( const MinFilterFtor &ftor ): _radius( ftor._radius ), _lastRow( TypeTraits< int32 >::Min ), _size( (2*ftor._radius+1) )
	{ _array = new ElType[ _size*_size ]; }

	~MinFilterFtor()
	{ delete [] _array; }

	template< typename Accessor >
	ElType
	Apply( const Vector< int32, 2 > &pos, Accessor &accessor ) 
	{
		Vector< int32, 2 > idx;
		unsigned i = 0;
		if( _lastRow == pos[1] ) {
			idx[0] = pos[0] + _radius;
			for( idx[1] = pos[1] - _radius; idx[1] <= pos[1] + _radius; ++idx[1] ) {
				_array[ i*_size + _lastCol ] = accessor( idx );
				++i;
			}
			_lastCol = (_lastCol + 1) % _size;
		} else {
			for( idx[1] = pos[1] - _radius; idx[1] <= pos[1] + _radius; ++idx[1] ) {
				for( idx[0] = pos[0] - _radius; idx[0] <= pos[0] + _radius; ++idx[0] ) {
					_array[i++] = accessor( idx );
				}
			}
			_lastCol = 0;
			_lastRow = pos[1];
		}
		//std::nth_element( &(_array[ 0 ]), &(_array[ _size*_size - 1 ]), &(_array[ _size*_size ]) );
		//return _array[ _size*_size - 1 ];
		
		return *std::min_element( &(_array[ 0 ]), &(_array[ _size*_size ]) );
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
MinFilter2D< InputImageType >
::MinFilter2D() : PredecessorType( new Properties() )
{

}

template< typename InputImageType >
MinFilter2D< InputImageType >
::MinFilter2D( typename MinFilter2D< InputImageType >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename InputImageType >
void
MinFilter2D< InputImageType >
::BeforeComputation( APipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	if( utype != APipeFilter::RECALCULATION 
		&& this->_propertiesTimestamp != GetProperties().GetTimestamp() )
	{
		utype = APipeFilter::RECALCULATION;
	}
}

template< typename InputImageType >
bool
MinFilter2D< InputImageType >
::Process2D(
			const typename MinFilter2D< InputImageType >::Region	&inRegion,
			typename MinFilter2D< InputImageType >::Region 	&outRegion
		 )
{
	if( !this->CanContinue() ) {
		return false;
	}

	try {
		MinFilterFtor< InputElementType > filter( GetRadius() );
		FilterProcessor< 
			MinFilterFtor< InputElementType >,
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


#endif /*_MIN_FILTER_H*/


