/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AGeometryDataset2D.h 
 * @{ 
 **/

#ifndef _AGEOMETRY_DATA_SET_2D_H
#define _AGEOMETRY_DATA_SET_2D_H

#include "MedV4D/Imaging/AGeometryDataset.h"
#include <vector>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/GeometryDatasetFactory.h"
#include "MedV4D/Imaging/ModificationManager.h"
#include "MedV4D/Imaging/GeometricalObject.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class AGeometryDataset2D: public AGeometryDataset
{
public:
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AGeometryDataset2D );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AGeometryDataset );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

protected:
	AGeometryDataset2D( DatasetType datasetType ): AGeometryDataset( datasetType ) 
		{}
};


class ASlicedGeometry: public AGeometryDataset2D
{
public:
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( ASlicedGeometry );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AGeometryDataset2D );
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

	virtual Geometry::GeometryTypeID
	GetSlicedGeometryObjectType()const = 0;
protected:
	ASlicedGeometry( int32 minSlice, int32 maxSlice ): AGeometryDataset2D( DATASET_SLICED_GEOMETRY ), _minSlice( minSlice ), _maxSlice( maxSlice )
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
	friend class GeometryDatasetFactory;

	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( SlicedGeometry< OType > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( ASlicedGeometry );
	PREPARE_CAST_METHODS_MACRO;
	IS_CONSTRUCTABLE_MACRO;

	typedef OType						ObjectType;
	typedef typename ObjectType::PointType			PointType;
	typedef typename ObjectType::Ptr			ObjectTypePtr;
	typedef std::vector< ObjectTypePtr > 			ObjectsInSlice;
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
	GetObjectInSlice( int32 sliceNumber, uint32 objectNumber )
		{
			ObjectsInSlice &slice = GetSlice( sliceNumber );
			if( objectNumber >= slice.size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			return *(slice[ objectNumber ]);
		}

	const ObjectType &
	GetObjectInSlice( int32 sliceNumber, uint32 objectNumber )const
		{
			const ObjectsInSlice &slice = GetSlice( sliceNumber );
			if( objectNumber >= slice.size() ) {
				_THROW_ M4D::ErrorHandling::EBadIndex( "Wrong object in slice index" );
			}
			return *(slice[ objectNumber ]);
		}

	uint32
	AddObjectToSlice( int32 sliceNumber, const ObjectType &obj )
		{
			ObjectsInSlice &slice = GetSlice( sliceNumber );
			ObjectType *o = new ObjectType( obj );

			slice.push_back( ObjectTypePtr( o ) );
			return slice.size();
		}

	uint32
	AddObjectToSlice( int32 sliceNumber, ObjectTypePtr obj )
		{
			ObjectsInSlice &slice = GetSlice( sliceNumber );
			slice.push_back( obj );
			return slice.size();
		}

	void
	RemoveObjectFromSlice( int32 sliceNumber, uint32 objectNumber )
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

	Geometry::GeometryTypeID
	GetSlicedGeometryObjectType() const
		{ return OType::GetGeometryObjectTypeID(); }

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
/** @} */

#endif /*_AGEOMETRY_DATA_SET_2D_H*/


