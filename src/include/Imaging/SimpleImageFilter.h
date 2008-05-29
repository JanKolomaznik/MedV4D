#ifndef _SIMPLE_IMAGE_FILTER_H
#define _SIMPLE_IMAGE_FILTER_H

#include "ExceptionBase.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"


namespace M4D
{

namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
class SimpleImageFilter: public AbstractPipeFilter
{
public:
	typedef typename InputPortImageFilter< InputImageType >
		InputPortType;
	typedef typename OutputPortImageFilter< OutputImageType >	
		OutputPortType;

	SimpleImageFilter();
protected:
	bool
	ExecutionThreadMethod();

	bool
	ExecutionOnWholeThreadMethod();

	const InputImageType&
	GetInputImage()const;

	OutputImageType&
	GetOutputImage()const;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( SimpleImageFilter );
};



/**
 * Ancestor of filters with single input and single output. 
 * Working on three dimensional data, but computation in different
 * slices is independent.
 ** /
template< typename InputElementType, typename OutputElementType >
class SimpleSlicedImageFilter: public Simple3DImageFilter< InputElementType, OutputElementType >
{
public:
	SimpleSlicedImageFilter();
protected:

private:

};
*/

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/SimpleImageFilter.h"

#endif /*_SIMPLE_IMAGE_FILTER_H*/
