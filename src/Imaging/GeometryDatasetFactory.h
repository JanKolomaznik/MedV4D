#ifndef _GEOMETRY_DATASET_FACTORY_H
#define _GEOMETRY_DATASET_FACTORY_H

#include "common/Common.h"
#include "Imaging/ADataset.h"
#include "AGeometryDataset2D.h"
#include "Imaging/DatasetSerializationTools.h"
#include "Imaging/GeometricalObject.h"

#include <iostream>
#include <fstream>
#include <string>

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometryDatasetFactory.h 
 * @{ 
 **/

namespace M4D
{

namespace Imaging
{

class GeometryDatasetFactory
{
public:


	template< typename OType >
	static typename SlicedGeometry< OType >::Ptr
	CreateSlicedGeometry( int32 minSlice, int32 maxSlice )
	{
		SlicedGeometry< OType > *geometry = new SlicedGeometry< OType >( minSlice, maxSlice );
		return typename SlicedGeometry< OType >::Ptr( geometry );
	}

	template< typename OType >
	static void
	ChangeSliceCount( SlicedGeometry< OType > &dataset, int32 minSlice, int32 maxSlice )
	{
		dataset.Resize( minSlice, maxSlice );
	}

	static ASlicedGeometry::Ptr
	DeserializeSlicedGeometryFromStream( M4D::IO::InStream &stream );

	static void
	DeserializeSlicedGeometryFromStream( M4D::IO::InStream &stream, ASlicedGeometry &geometry );

	static void 
	SerializeSlicedGeometry(M4D::IO::OutStream &stream, const ASlicedGeometry &dataset);

	template< typename OType >
	static void 
	SerializeSlicedGeometry(M4D::IO::OutStream &stream, const SlicedGeometry< OType > &dataset)
	{
		SerializeHeader( stream, DATASET_SLICED_GEOMETRY );

		stream.Put<uint32>( dataset.GetSlicedGeometryObjectType() );

		stream.Put<int32>( dataset.GetSliceMin() );
		stream.Put<int32>( dataset.GetSliceMax() );

		stream.Put<uint32>( DUMP_HEADER_END_MAGIC_NUMBER );

		for( int32 i = dataset.GetSliceMin(); i < dataset.GetSliceMax(); ++i ) {
			stream.Put<uint32>( DUMP_SLICE_BEGIN_MAGIC_NUMBER );
			const typename SlicedGeometry< OType >::ObjectsInSlice	&slice = dataset.GetSlice( i );
			stream.Put<uint32>( slice.size() );

				std::for_each( slice.begin(), slice.end(), M4D::Imaging::Geometry::SerializeGeometryObjectFtor< OType >( stream ) );

			stream.Put<uint32>( DUMP_SLICE_END_MAGIC_NUMBER );
		}
	}

protected:

};



} /*namespace Imaging*/
/** @} */
} /*namespace M4D*/

#endif //_GEOMETRY_DATASET_FACTORY_H
