#ifndef IMAGE_REGION_ACCESSORS_H
#define IMAGE_REGION_ACCESSORS_H

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

	SimpleAccessor( const RegionType & region ): PredecessorType( region )
	{}

	typename RegionType::ElementType
	operator()( const typename PredecessorType::CoordType &pos )const
	{
		return this->_region.GetElementFast( pos );
	}

};

template< typename RegionType >
class MirrorAccessor: public AccessorBase< RegionType >
{
public:
	typedef AccessorBase< RegionType > PredecessorType;

	MirrorAccessor( const RegionType & region ): PredecessorType( region ), _minimum( region.GetMinimum() )
	{
		_maximumIn = this->_region.GetMaximum() - typename PredecessorType::CoordType( 1 );
	}

	typename RegionType::ElementType
	operator()( const typename PredecessorType::CoordType &pos )const
	{
		if( pos >= _minimum && pos <= _maximumIn ) {
				//return this->_region.GetElement( pos );
				return this->_region.GetElementFast( pos );
		}
		typename PredecessorType::CoordType tmp = pos - _minimum;
		VectorAbs( tmp );
		tmp += _minimum;

		tmp = _maximumIn - tmp;
		VectorAbs( tmp );
		tmp = _maximumIn - tmp;

		return this->_region.GetElement( tmp );
	}

	typename PredecessorType::CoordType	_minimum;
	typename PredecessorType::CoordType	_maximumIn;
};



} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*IMAGE_REGION_ACCESSORS_H*/
