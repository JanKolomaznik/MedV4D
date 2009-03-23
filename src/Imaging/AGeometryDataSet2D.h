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
#include "common/Common.h"
#include "Imaging/GeometryDataSetFactory.h"
#include "Imaging/ModificationManager.h"

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
	
	WriterBBoxInterface &
	SetDirtyBBox( 
			int32 min,
			int32 max 
			)
		{ return _modificationManager.AddMod( Vector< int32, 1 >( min ), Vector< int32, 1 >( max ) ); }

	WriterBBoxInterface &
	SetWholeDirtyBBox()
		{ return SetDirtyBBox( _minSlice, _maxSlice ); }


	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			int32 min,
			int32 max 
			)const
		{ return _modificationManager.GetMod( Vector< int32, 1 >( min ), Vector< int32, 1 >( max ) ); }

	ReaderBBoxInterface::Ptr 
	GetWholeDirtyBBox()const
		{ return GetDirtyBBox( _minSlice, _maxSlice ); }
	
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

	mutable ModificationManager _modificationManager;
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
				_THROW_ M4D::ErrorHandling::EBadIndex( 
						TO_STRING( "Wrong slice index = " << idx << "; minSlice = " << this->_minSlice << "; maxSlice = " << this->_maxSlice )
							);
			}
			return _slices[ idx - this->_minSlice ];
		}

	const ObjectsInSlice &
	GetSlice( int32 idx )const
		{ 
			if ( idx < this->_minSlice || idx >= this->_maxSlice ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( 
						TO_STRING( "Wrong slice index = " << idx << "; minSlice = " << this->_minSlice << "; maxSlice = " << this->_maxSlice )
							);
			}
			return _slices[ idx - this->_minSlice ];
		}
	uint32
	GetSliceSize( int32 idx )const
		{
			return GetSlice( idx ).size();
		}

	ObjectType &
	GetObject( int32 sliceNumber, uint32 objectNumber )
		{
			ObjectsInSlice &slice = GetSlice( sliceNumber );
			if( objectNumber >= slice.size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			return slice[ objectNumber ];
		}

	const ObjectType &
	GetObject( int32 sliceNumber, uint32 objectNumber )const
		{
			const ObjectsInSlice &slice = GetSlice( sliceNumber );
			if( objectNumber >= slice.size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			return slice[ objectNumber ];
		}

	uint32
	AddObject( int32 sliceNumber, const ObjectType &obj )
		{
			ObjectsInSlice &slice = GetSlice( sliceNumber );
			slice.push_back( obj );
			return slice.size();
		}

	void
	RemoveObject( int32 sliceNumber, uint32 objectNumber )
		{
			ObjectsInSlice &slice = GetSlice( sliceNumber );
			if( objectNumber >= slice.size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			slice.erase( slice.begin() + objectNumber );
		}



	

	void Dump( void)
	{ _THROW_ ErrorHandling::ETODO();	}
	void SerializeData(M4D::IO::OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void SerializeClassInfo(M4D::IO::OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void SerializeProperties(M4D::IO::OutStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void DeSerializeData(M4D::IO::InStream &stream)
	{ _THROW_ ErrorHandling::ETODO();	}
	void DeSerializeProperties(M4D::IO::InStream &stream)
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


