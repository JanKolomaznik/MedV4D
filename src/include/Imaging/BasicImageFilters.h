#ifndef _BASIC_IMAGE_FILTERS_H
#define _BASIC_IMAGE_FILTERS_H

#include "Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/SimpleImageFilter.h"


namespace M4D
{

namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
class CopyImageFilter: public ImageFilter< InputImageType, OutputImageType >
{
public:
	CopyImageFilter();
	~CopyImageFilter(){}
protected:
	bool
	ExecutionThreadMethod();

	bool
	ExecutionOnWholeThreadMethod();

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
