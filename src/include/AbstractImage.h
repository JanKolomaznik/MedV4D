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
	typedef boost::shared_ptr< AbstractImage > APtr;


	virtual ~AbstractImage()=0;

	virtual int
	GetElementTypeID()=0;

	virtual size_t
	GetElementCount()=0;
};


} /*namespace Images*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_H*/
