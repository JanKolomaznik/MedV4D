#ifndef _ABSTRACT_IMAGE_FILTER_H
#define _ABSTRACT_IMAGE_FILTER_H

#include "ExceptionBase.h"

#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"

namespace M4D
{

namespace Imaging
{


/**
 * Ancestor of filters with single input and single output.
 **/
template< typename InputElementType, typename OutputElementType >
class SimpleImageFilter: public AbstractFilter
{
public:

protected:

private:

};

} /*namespace Imaging*/
} /*namespace M4D*/


#endif /*_ABSTRACT_IMAGE_FILTER_H*/
