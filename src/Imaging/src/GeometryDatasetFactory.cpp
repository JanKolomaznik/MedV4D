#include "common/Common.h"

#include "Imaging/GeometryDatasetFactory.h"

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file GeometryDatasetFactory.cpp
 * @{ 
 **/

namespace M4D
{

namespace Imaging
{

ASlicedGeometry::Ptr
GeometryDatasetFactory::DeserializeSlicedGeometryFromStream( M4D::IO::InStream &stream ) 
{
	return ASlicedGeometry::Ptr(); 
}

void
GeometryDatasetFactory::DeserializeSlicedGeometryFromStream( M4D::IO::InStream &stream, ASlicedGeometry &geometry ) 
{

}

void 
GeometryDatasetFactory::SerializeSlicedGeometry(M4D::IO::OutStream &stream, const ASlicedGeometry &dataset)
{

}


} /*namespace Imaging*/
/** @} */
} /*namespace M4D*/
