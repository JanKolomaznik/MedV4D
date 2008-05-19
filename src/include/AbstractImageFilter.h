#ifndef _ABSTRACT_IMAGE_FILTER_H
#define _ABSTRACT_IMAGE_FILTER_H

#include "ExceptionBase.h"

#include "ImageDataTemplate.h"
#include "ImageFactory.h"

namespace M4D
{

namespace Images
{

/**
 * Ancestor of all image filters.
 **/
class BaseFilter
{

};


/**
 * Ancestor of filters with single input and single output.
 **/
template< typename InputElementType, typename OutputElementType >
class AbstractImageDataFilter: public BaseFilter
{
public:

protected:

private:

};

} /*namespace Images*/
} /*namespace M4D*/


#endif /*_ABSTRACT_IMAGE_FILTER_H*/
