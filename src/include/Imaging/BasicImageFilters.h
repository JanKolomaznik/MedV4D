#ifndef _BASIC_IMAGE_FILTERS_H
#define _BASIC_IMAGE_FILTERS_H

#include "Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractImageFilters.h"


namespace M4D
{

namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
class CopyImageFilter: public ImageSliceFilter< InputImageType, OutputImageType >
{
public:
	CopyImageFilter();
	~CopyImageFilter(){}
protected:
	void
	ProcessSlice(
			const InputImageType 	&in,
			OutputImageType		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    );

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( CopyImageFilter );
};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/BasicImageFilters.tcc"

#endif /*_BASIC_IMAGE_FILTERS_H*/
