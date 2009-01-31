/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImageSliceFilter.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_SLICE_FILTER_H
#define _ABSTRACT_IMAGE_SLICE_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageFilter.h"
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

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
class AbstractImageSliceFilter;

template< typename InputElementType, typename OutputImageType >
class AbstractImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
	 : public AbstractImageFilter< Image< InputElementType, 3 >, OutputImageType >
{
public:
	typedef AbstractImageFilter< Image< InputElementType, 3 >, OutputImageType >	PredecessorType;
	
	struct Properties : public PredecessorType::Properties
	{
		Properties( unsigned sliceNeighbourCount, unsigned computationGrouping ) 
			: _sliceComputationNeighbourCount( sliceNeighbourCount ), _computationGrouping( computationGrouping ) {}
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
	
	AbstractImageSliceFilter( Properties *prop );
	~AbstractImageSliceFilter() {}

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
		       		_THROW_ ErrorHandling::ExceptionBadParameter< unsigned >( count );
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
			int32					x1,
			int32					y1,
			int32					x2,
			int32					y2,
			int32					slice
		    ) = 0;


	/*virtual bool
	ProcessSlice(
			const ImageRegion< InputImageType, 3 >		&in,
			OutputImageType					&out,
			int32						slice
		    ) = 0;

	virtual bool
	ProcessSlice(
			const ImageRegion< InputImageType, 3 >		&in,
			const ImageRegion< OutputElementType, 2 > 	&out,
			int32						slice
		    ) = 0;*/

	virtual WriterBBoxInterface &
	GetComputationGroupWriterBBox( SliceComputationRecord & record ) = 0;

	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE /*utype*/ );


	void
	MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype );

	

	ComputationGroupList		_actualComputationGroups;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageSliceFilter );
};


/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType >
class AbstractImageSliceFilterIExtents;

template< typename InputElementType, typename OutputElementType >
class AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
	 : public AbstractImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
	typedef typename Imaging::AbstractImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;
	
	typedef typename PredecessorType::Properties	Properties;

	AbstractImageSliceFilterIExtents( Properties *prop );
	~AbstractImageSliceFilterIExtents() {}

protected:

	WriterBBoxInterface &
	GetComputationGroupWriterBBox( SliceComputationRecord & record );

	void
	PrepareOutputDatasets();

	/*bool
	ProcessSlice(
			const ImageRegion< InputImageType, 3 >		&in,
			const ImageRegion< OutputElementType, 2 > 	&out,
			int32						slice
		    ) = 0;*/
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageSliceFilterIExtents );
};


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/AbstractImageSliceFilter.tcc"

#endif /*_ABSTRACT_IMAGE_SLICE_FILTER_H*/

/** @} */

