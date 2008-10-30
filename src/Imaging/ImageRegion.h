#ifndef _IMAGE_REGION_H
#define _IMAGE_REGION_H

#include "Imaging/ImageIterator.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageRegion.h 
 * @{ 
 **/

namespace Imaging
{




template< typename ElementType, uint32 Dim >
class ImageRegion
{
public:
	static const uint32 Dimension = Dim;
	typedef ImageIterator	Iterator;

	Iterator
	GetIterator()const;

	ImageRegion
	Intersection( const ImageRegion & region );

	ImageRegion
	UnionBBox( const ImageRegion & region );

	uint32
	GetSize( unsigned dim )const;

	int32
	GetCoordinates( unsigned dim )const;

	GetElement( 
protected:
	
private:
	ElementType	*_pointer;
	uint32		_size[ Dimension ];
	int32		_strides[ Dimension ];
	int32		_coordinates[ Dimension ];
};


}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_IMAGE_REGION_H*/
