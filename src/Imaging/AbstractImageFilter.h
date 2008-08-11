#ifndef _ABSTRACT_IMAGE_FILTER_H
#define _ABSTRACT_IMAGE_FILTER_H

#include "Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImagePorts.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/ModificationManager.h"
#include "Imaging/ImageTraits.h"

#include <vector>

namespace M4D
{

namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
class AbstractImageFilter: public AbstractPipeFilter
{
public:
	typedef AbstractPipeFilter	PredecessorType;

	typedef typename M4D::Imaging::ImageTraits< InputImageType >::InputPort
		InputPortType;
	typedef typename M4D::Imaging::ImageTraits< OutputImageType >::OutputPort
		OutputPortType;
	typedef InputImageType	InputImage;
	typedef OutputImageType OutputImage;

	struct Properties : public PredecessorType::Properties
	{

	};

	~AbstractImageFilter() {}
protected:
	AbstractImageFilter( Properties * prop );

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
	PrepareOutputDatasets();

	void
	AfterComputation( bool successful );


	const InputImage	*in;
	Common::TimeStamp	_inTimestamp;
	Common::TimeStamp	_inEditTimestamp;


	OutputImage		*out;
	Common::TimeStamp	_outTimestamp;
	Common::TimeStamp	_outEditTimestamp;

	
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageFilter );
};

/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType >
class ImageVolumeFilter;

template< typename InputElementType, typename OutputImageType >
class ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
	 : public AbstractImageFilter< Image< InputElementType, 3 >, OutputImageType >
{
public:
	typedef typename Imaging::AbstractImageFilter< Image< InputElementType, 3 >, OutputImageType >	PredecessorType;

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
	 : public AbstractImageFilter< Image< InputElementType, 4 >, OutputImageType >
{
public:
	typedef typename Imaging::AbstractImageFilter< Image< InputElementType, 4 >, OutputImageType >	PredecessorType;

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
#include "Imaging/AbstractImageFilter.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_H*/
