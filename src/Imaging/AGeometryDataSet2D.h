/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AGeometryDataSet2D.h 
 * @{ 
 **/

#ifndef _AGEOMETRY_DATA_SET_2D_H
#define _AGEOMETRY_DATA_SET_2D_H

#include "Imaging/AGeometryDataSet.h"
#include <vector>
#include "Common.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class AGeometryDataSet2D: public AGeometryDataSet
{
public:

protected:
	AGeometryDataSet2D( DataSetType datasetType ): AGeometryDataSet( datasetType ) 
		{}
};


class ASlicedGeometry: public AGeometryDataSet2D
{
public:
	
	int32
	GetSliceMin()const
		{ return _minSlice; }

	int32
	GetSliceMax()const
		{ return _maxSlice; }
protected:
	ASlicedGeometry( int32 minSlice, int32 maxSlice ): AGeometryDataSet2D( DATASET_SLICED_GEOMETRY ), _minSlice( minSlice ), _maxSlice( maxSlice )
		{}

	int32 _minSlice;
	int32 _maxSlice;
};

//TODO - locking

template< 
	typename CoordType, 
	template< typename CType, unsigned Dim > class OType 
	>
class SlicedGeometry: public ASlicedGeometry
{
public:
	typedef SlicedGeometry< CoordType, OType >		ThisClass;
	typedef boost::shared_ptr< ThisClass >			Ptr;
	typedef OType< CoordType, 2 >				ObjectType;
	typedef typename ObjectType::PointType			PointType;
	typedef std::vector< ObjectType > 			ObjectsInSlice;
	typedef std::vector< ObjectsInSlice >			Slices;

	SlicedGeometry( int32 minSlice, int32 maxSlice ): ASlicedGeometry( minSlice, maxSlice )
		{
			_slices.resize( maxSlice - minSlice );
		}

	ObjectsInSlice &
	GetSlice( int32 idx )
		{ 
			if ( idx < this->_minSlice || idx >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong slice index" );
			}
			return _slices[ idx - this->_minSlice ];
		}

	const ObjectsInSlice &
	GetSlice( int32 idx )const
		{ 
			if ( idx < this->_minSlice || idx >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong slice index" );
			}
			return _slices[ idx - this->_minSlice ];
		}
	uint32
	GetSliceSize( int32 idx )const
		{
			if ( idx < this->_minSlice || idx >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong slice index" );
			}
			return _slices[ idx - this->_minSlice ].size();
		}

	ObjectType &
	GetObject( int32 sliceNumber, uint32 objectNumber )
		{
			if ( sliceNumber < this->_minSlice || sliceNumber >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong slice index" );
			}
			if( objectNumber >= _slices[ sliceNumber - this->_minSlice ].size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			return _slices[ sliceNumber - this->_minSlice ][ objectNumber ];
		}

	const ObjectType &
	GetObject( int32 sliceNumber, uint32 objectNumber )const
		{
			if ( sliceNumber < this->_minSlice || sliceNumber >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong slice index" );
			}
			if( objectNumber >= _slices[ sliceNumber - this->_minSlice ].size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			return _slices[ sliceNumber - this->_minSlice ][ objectNumber ];
		}

	uint32
	AddObject( int32 sliceNumber, const ObjectType &obj )
		{
			if ( sliceNumber < this->_minSlice || sliceNumber >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong slice index" );
			}
			_slices[ sliceNumber - this->_minSlice ].push_back( obj );
			return _slices[ sliceNumber - this->_minSlice ].size();
		}



	void Dump( void)
	{ _THROW_ ErrorHandling::ETODO();	}

	void Serialize(OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void DeSerialize(InStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
protected:

	Slices	_slices;
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_AGEOMETRY_DATA_SET_2D_H*/


