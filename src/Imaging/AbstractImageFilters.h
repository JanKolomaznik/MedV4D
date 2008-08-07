#ifndef _ABSTRACT_IMAGE_FILTERS_H
#define _ABSTRACT_IMAGE_FILTERS_H

#include "Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImagePorts.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/ModificationManager.h"

#include <vector>

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

	struct Properties : public AbstractPipeFilter::Properties
	{

	};

	~ImageFilter() {}
protected:
	ImageFilter( Properties * prop );

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
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageFilter );
};

struct SliceComputationRecord
{
	ReaderBBoxInterface::Ptr	inputBBox;
	WriterBBoxInterface		*writerBBox;
	int32				firstSlice;
	int32				lastSlice;
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
	
	struct Properties : public PredecessorType::Properties
	{
		/**
		 * How many slices to up and down are needed for computation.
		 * This information is needed when waiting for input update.
		 **/
		unsigned	_sliceComputationNeighbourCount;

		/**
		 * How many slices will be put into one computation sequence.
		 **/
		unsigned	_computationGrouping;
	};
	
	ImageSliceFilter( Properties *prop );
	~ImageSliceFilter() {}

	void
	SetComputationNeighbourhood( unsigned count )
		{ static_cast<Properties*>(this->_properties)->_sliceComputationNeighbourCount = count; }

	unsigned
	GetComputationNeighbourhood()
		{ return static_cast<Properties*>(this->_properties)->_sliceComputationNeighbourCount; }

	void
	SetComputationGrouping( unsigned count )
		{ 
			if( count > 0 ) {
				static_cast<Properties*>(this->_properties)->_computationGrouping = count; 
			}else {
		       		throw ErrorHandling::ExceptionBadParameter< unsigned >( count );
			}	
		}

	unsigned
	GetComputationGrouping()
		{ return static_cast<Properties*>(this->_properties)->_computationGrouping; }
protected:

	typedef std::vector< SliceComputationRecord >	ComputationGroupList;

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

	virtual WriterBBoxInterface &
	GetComputationGroupWriterBBox( SliceComputationRecord & record ) = 0;

	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	
	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	

	

	ComputationGroupList		_actualComputationGroups;
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
	
	struct Properties : public PredecessorType::Properties
	{

	};

	IdenticalExtentsImageSliceFilter( Properties *prop );
	~IdenticalExtentsImageSliceFilter() {}

protected:

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	WriterBBoxInterface &
	GetComputationGroupWriterBBox( SliceComputationRecord & record );

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

template< typename InputImageType, typename OutputImageType >
class ImageFilterWholeAtOnce 
	: public ImageFilter< InputImageType, OutputImageType >
{
public:
	typedef ImageFilter< InputImageType, OutputImageType >	PredecessorType;
	typedef typename PredecessorType::Properties		Properties;

	ImageFilterWholeAtOnce( Properties *prop );
protected:

	virtual bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    ) = 0;

	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	
	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	AfterComputation( bool successful );

	ReaderBBoxInterface::Ptr	_readerBBox;
	WriterBBoxInterface		*_writerBBox;

private:
	ReaderBBoxInterface::Ptr
	ApplyReaderBBox( const InputImageType &in );

	WriterBBoxInterface *
	ApplyWriterBBox( OutputImageType &out );

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractImageFilters.tcc"

#endif /*_ABSTRACT_IMAGE_FILTERS_H*/
