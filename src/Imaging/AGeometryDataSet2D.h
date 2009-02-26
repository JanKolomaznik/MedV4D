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
#include "Imaging/GeometryDataSetFactory.h"

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
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AGeometryDataSet2D );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AGeometryDataSet );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

protected:
	AGeometryDataSet2D( DataSetType datasetType ): AGeometryDataSet( datasetType ) 
		{}
};


class ASlicedGeometry: public AGeometryDataSet2D
{
public:
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( ASlicedGeometry );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AGeometryDataSet2D );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;
	
	
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

template< typename OType >
class SlicedGeometry: public ASlicedGeometry
{
public:
	friend class GeometryDataSetFactory;

	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( SlicedGeometry< OType > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( ASlicedGeometry );
	PREPARE_CAST_METHODS_MACRO;
	IS_CONSTRUCTABLE_MACRO;

	typedef OType						ObjectType;
	typedef typename ObjectType::PointType			PointType;
	typedef std::vector< ObjectType > 			ObjectsInSlice;
	typedef std::vector< ObjectsInSlice >			Slices;

	SlicedGeometry( int32 minSlice, int32 maxSlice ): ASlicedGeometry( minSlice, maxSlice )
		{
			_slices.resize( maxSlice - minSlice );
		}
	SlicedGeometry(): ASlicedGeometry( 0, 0 )
		{ }

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
	void SerializeData(OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void SerializeClassInfo(OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void SerializeProperties(OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void DeSerializeData(InStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void DeSerializeProperties(InStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}

protected:
	void
	Resize( int32 minSlice, int32 maxSlice )
	{
		//TODO check values
		_minSlice = minSlice;
		_maxSlice = maxSlice;
		_slices.resize( maxSlice - minSlice );
	}

	Slices	_slices;
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_AGEOMETRY_DATA_SET_2D_H*/


