#ifndef _GEOMETRY_DATASET_FACTORY_H
#define _GEOMETRY_DATASET_FACTORY_H

#include "common/Common.h"
#include "Imaging/AbstractDataSet.h"
#include "AGeometryDataSet2D.h"

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

class GeometryDataSetFactory
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

protected:

};



} /*namespace Imaging*/
/** @} */
} /*namespace M4D*/

#endif //_GEOMETRY_DATASET_FACTORY_H
