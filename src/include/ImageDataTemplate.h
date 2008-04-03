#ifndef _IMAGE_DATA_TEMPLATE_H
#define _IMAGE_DATA_TEMPLATE_H

#include "AbstractImage.h"

namespace Images
{

//Forward declaration
class ImageFactory;

/**
 * Structure containing information about image in one dimension. Each 
 * dimension is supposed to have its own informations structure.
 **/
struct DimensionInformations
{
	/**
	 * Width of image in actual dimension.
	 **/
	uint32		size;
	/**
	 * Stride, which is used to increase coordinates in actual dimension.
	 **/
	uint32		stride;
};


template < typename ElementType >
class ImageDataTemplate
{
public:
	friend ImageFactory;
	typedef typename boost::shared_ptr< ImageDataTemplate< ElementType > Ptr;

	~ImageDataTemplate();	
protected:

private:

	unsigned		_dimension;
	DimensionInformations	*_parameters;
	ElementType		*_data;
};



} /*namespace Images*/

/*Include template implementation.*/
#include "ImageDataTemplate.tcc"


#endif /*_IMAGE_DATA_TEMPLATE_H*/
