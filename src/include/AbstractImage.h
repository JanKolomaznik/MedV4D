#ifndef _ABSTRACT_IMAGE_H
#define _ABSTRACT_IMAGE_H

#include "M4DCommon.h"

#include <boost/shared_ptr.hpp>

namespace M4D
{

namespace Images
{

class AbstractImage
{
public:
	/**
	 * Smart pointer type for accesing AbstractImage instance (child).
	 **/
	typedef boost::shared_ptr< AbstractImage > Ptr;


	virtual ~AbstractImage()=0;
};


} /*namespace Images*/
} /*namespace M4D*/
#endif /*_ABSTRACT_IMAGE_H*/
