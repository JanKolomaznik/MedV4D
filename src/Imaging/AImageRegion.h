#ifndef _AIMAGE_REGION_H
#define _AIMAGE_REGION_H

#include "common/Common.h"
#include "common/Vector.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImageRegion.h 
 * @{ 
 **/

namespace Imaging
{


class AImageRegion
{


public:
	virtual ~AImageRegion() {}

	virtual uint32 
	GetDimension()const = 0;

	virtual int16 
	GetElementTypeID()const = 0;
};

template< unsigned Dim >
class AImageRegionDim: public AImageRegion
{
public:
	typedef Vector< int, Dim >	PointType;
	static const unsigned Dimension = Dim;


	unsigned 
	GetDimension()const
		{ return Dimension; }

	unsigned
	GetSize( unsigned dim )const
		{
			return _size[dim];
		}

	Vector< unsigned, Dimension >
	GetSize()const
		{
			return _size;
		}

	int
	GetMinimum( unsigned dim )const
		{
			return _origin[dim];
		}

	PointType
	GetMinimum()const
		{
			return _origin;
		}

	int
	GetMaximum( unsigned dim )const
		{
			return _origin[dim] + _size[dim];
		}

	PointType
	GetMaximum()const
		{
			return _origin + PointType( (int32*)_size.GetData() );
		}

	Vector< float, Dimension >
	GetElementExtents()const
		{ return this->_elementExtents; }

	Vector< float, Dimension >
	GetRealMinimum()const
		{
			return VectorMemberProduct( _origin, _elementExtents );
		}

	Vector< float, Dimension >
	GetRealMaximum()const
		{
			return VectorMemberProduct( GetMaximum(), _elementExtents );
		}

	Vector< float, Dimension >
	GetRealSize()const
		{
			return VectorMemberProduct( _size, _elementExtents );
		}

protected:
	AImageRegionDim(
			const Vector< unsigned, Dimension >	&size = Vector< unsigned, Dimension >( 0 ),
			const Vector< int, Dimension >	&origin = Vector< int, Dimension >( 0 ),
			const Vector< float, Dimension >	&elementExtents = Vector< float, Dimension >( 1.0f )
		       ): _elementExtents( elementExtents ), _size( size ), _origin( origin )
	{}
	
	AImageRegionDim( const AImageRegionDim &region )
		: _elementExtents( region._elementExtents ), _size( region._size ), _origin( region._origin )
	{}

	Vector< float, Dimension >	_elementExtents;
	Vector< unsigned, Dimension >	_size;
	Vector< int, Dim >		_origin;

};

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_AIMAGE_REGION_H*/
