#ifndef _ABSTRACT_IMAGE_FILTERS_H
#define _ABSTRACT_IMAGE_FILTERS_H

#include "Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"


namespace M4D
{

namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
class ImageFilter: public AbstractPipeFilter
{
public:
	typedef typename M4D::Imaging::InputPortImageFilter< InputImageType >
		InputPortType;
	typedef typename M4D::Imaging::OutputPortImageFilter< OutputImageType >	
		OutputPortType;
	typedef InputImageType	InputImage;
	typedef OutputImageType OutputImage;

	ImageFilter();
	~ImageFilter() {}
protected:
	/* Must be reimplemented in successor.
	bool
	ExecutionThreadMethod();

	bool
	ExecutionOnWholeThreadMethod();*/

	const InputImageType&
	GetInputImage()const;

	OutputImageType&
	GetOutputImage()const;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageFilter );
};

/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType >
class ImageSliceFilter;

template< typename InputElementType, typename OutputImageType >
class ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
	 : public ImageFilter< Image< InputElementType, 3 >, OutputImageType >
{
public:

	ImageSliceFilter();
	~ImageSliceFilter() {}
protected:
	/**
	 * This method should be overridden in successor. It is supposed to
	 * do calculation over one slice of input.
	 * \param in Input image.
	 * \param out Output image.
	 * \param x1 X coordinate of first point defining handled rectangle.
	 * \param y1 Y coordinate of first point defining handled rectangle.
	 * \param x2 X coordinate of second point defining handled rectangle.
	 * \param y2 Y coordinate of second point defining handled rectangle.
	 * \param slice Number of handled slice.
	 **/
	virtual void
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			OutputImageType				&out,
			size_t					x1,
			size_t					y1,
			size_t					x2,
			size_t					y2,
			size_t					slice
		    ) = 0;

	bool
	ExecutionThreadMethod();

	bool
	ExecutionOnWholeThreadMethod();
	
	void
	PrepareOutputDatasets();

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageSliceFilter );
};



} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractImageFilters.tcc"

#endif /*_ABSTRACT_IMAGE_FILTERS_H*/
