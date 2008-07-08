#ifndef _EXAMPLE_IMAGE_FILTERS_H
#define _EXAMPLE_IMAGE_FILTERS_H

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
class CopyImageFilter;

template< typename InputElementType, typename OutputElementType >
class CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
	CopyImageFilter();
	~CopyImageFilter(){}

protected:
	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< OutputElementType, 3 >		&out,
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

template< typename ElementType >
class ColumnMaxImageFilter
: public ImageVolumeFilter< Image< ElementType, 3 >, Image< ElementType, 2 > >
{
public:
	ColumnMaxImageFilter();
	~ColumnMaxImageFilter(){}

	void
	PrepareOutputDatasets();

protected:
	bool
	ProcessVolume(
			const Image< ElementType, 3 > 		&in,
			Image< ElementType, 2 >			&out,
			size_t					x1,
			size_t					y1,
			size_t					z1,
			size_t					x2,
			size_t					y2,
			size_t					z2
		    );
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ColumnMaxImageFilter );
};

template< typename InputImageType >
class SimpleThresholdingImageFilter;

template< typename InputElementType >
class SimpleThresholdingImageFilter< Image< InputElementType, 3 > >
: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:
	SimpleThresholdingImageFilter();
	~SimpleThresholdingImageFilter(){}

protected:
	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
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
	PROHIBIT_COPYING_OF_OBJECT_MACRO( SimpleThresholdingImageFilter );
};

template< typename InputImageType >
class SimpleConvolutionImageFilter;

template< typename InputElementType >
class SimpleConvolutionImageFilter< Image< InputElementType, 3 > >
: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:
	SimpleConvolutionImageFilter();
	~SimpleConvolutionImageFilter(){}

protected:
	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
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
	PROHIBIT_COPYING_OF_OBJECT_MACRO( SimpleConvolutionImageFilter );
};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/ExampleImageFilters.tcc"

#endif /*_EXAMPLE_IMAGE_FILTERS_H*/
