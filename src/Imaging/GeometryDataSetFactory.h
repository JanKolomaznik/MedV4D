#ifndef _GEOMETRY_DATASET_FACTORY_H
#define _GEOMETRY_DATASET_FACTORY_H

#include "Common.h"
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


	template< 
	typename CoordType, 
	template< typename CType, unsigned Dim > class OType 
	>
	static typename SlicedGeometry< CoordType, OType >::Ptr
	CreateSlicedGeometry( int32 minSlice, int32 maxSlice )
	{
		SlicedGeometry< CoordType, OType > *geometry = new SlicedGeometry< CoordType, OType >( minSlice, maxSlice );
		return typename SlicedGeometry< CoordType, OType >::Ptr( geometry );
	}

protected:

};



} /*namespace Imaging*/
/** @} */
} /*namespace M4D*/

#endif //_GEOMETRY_DATASET_FACTORY_H
