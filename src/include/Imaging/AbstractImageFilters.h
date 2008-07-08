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
	typedef AbstractPipeFilter	PredecessorType;

	typedef typename M4D::Imaging::InputPortImageFilter< InputImageType >
		InputPortType;
	typedef typename M4D::Imaging::OutputPortImageFilter< OutputImageType >	
		OutputPortType;
	typedef InputImageType	InputImage;
	typedef OutputImageType OutputImage;

	ImageFilter();
	~ImageFilter() {}
protected:

	const InputImageType&
	GetInputImage()const;

	void
	ReleaseInputImage()const;

	OutputImageType&
	GetOutputImage()const;

	void
	ReleaseOutputImage()const;

	void
	SetOutputImageSize( 
			size_t 		minimums[], 
			size_t 		maximums[], 
			float32		elementExtents[]
		    );

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	AfterComputation( bool successful );


	const InputImage	*in;
	OutputImage		*out;
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
	typedef typename Imaging::ImageFilter< Image< InputElementType, 3 >, OutputImageType >	PredecessorType;

	ImageSliceFilter();
	~ImageSliceFilter() {}

	void
	SetComputationNeighbourhood( unsigned count )
		{ _sliceComputationNeighbourCount = count; }

	unsigned
	GetComputationNeighbourhood()
		{ return _sliceComputationNeighbourCount; }
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
	 * \return True, if method finished its job without interrupting.
	 **/
	virtual bool
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
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	
	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	/**
	 * How many slices to up and down are needed for computation.
	 * This information is needed when waiting for input update.
	 **/
	unsigned	_sliceComputationNeighbourCount;

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageSliceFilter );
};


/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType >
class IdenticalExtentsImageSliceFilter;

template< typename InputElementType, typename OutputElementType >
class IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
	 : public ImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
	typedef typename Imaging::ImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;

	IdenticalExtentsImageSliceFilter();
	~IdenticalExtentsImageSliceFilter() {}

protected:

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	PrepareOutputDatasets();

private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( IdenticalExtentsImageSliceFilter );
};


/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType >
class ImageVolumeFilter;

template< typename InputElementType, typename OutputImageType >
class ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
	 : public ImageFilter< Image< InputElementType, 3 >, OutputImageType >
{
public:
	typedef typename Imaging::ImageFilter< Image< InputElementType, 3 >, OutputImageType >	PredecessorType;

	ImageVolumeFilter();
	~ImageVolumeFilter() {}
	
	void
	PrepareOutputDatasets();
protected:
	virtual bool
	ProcessVolume(
			const Image< InputElementType, 3 > 	&in,
			OutputImageType				&out,
			size_t					x1,
			size_t					y1,
			size_t					z1,
			size_t					x2,
			size_t					y2,
			size_t					z2
		    ) = 0;

	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );


	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageVolumeFilter );
};

template< typename InputElementType, typename OutputImageType >
class ImageVolumeFilter< Image< InputElementType, 4 >, OutputImageType >
	 : public ImageFilter< Image< InputElementType, 4 >, OutputImageType >
{
public:
	typedef typename Imaging::ImageFilter< Image< InputElementType, 4 >, OutputImageType >	PredecessorType;

	ImageVolumeFilter();
	~ImageVolumeFilter() {}

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );
protected:
	virtual bool
	ProcessVolume(
			const Image< InputElementType, 4 > 	&in,
			OutputImageType				&out,
			size_t					x1,
			size_t					y1,
			size_t					z1,
			size_t					x2,
			size_t					y2,
			size_t					z2,
			size_t					t
		    ) = 0;

	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	void
	PrepareOutputDatasets();

	
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageVolumeFilter );
};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractImageFilters.tcc"

#endif /*_ABSTRACT_IMAGE_FILTERS_H*/
