/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImageFilterWholeAtOnce.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H
#define _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H

#include "common/Common.h"
#include "Imaging/AImageFilter.h"
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

/**
 * This template is prepared for creation of image filters which need to access whole input dataset
 * or for experimental implementation - without progressive computing. 
 *
 * Before call of ProcessImage() filter waits on read bounding box with same proportion as image and after
 * that write bounding box containing output image is marked as dirty. After finished computation is this bounding box
 * marked as modified or cancelled if computation did not finished successfuly - ProcessImage() returned false.
 *
 * In classes inheriting from this one you must override methods ProcessImage() and PrepareOutputDatasets().
 **/
template< typename InputImageType, typename OutputImageType >
class AImageFilterWholeAtOnce 
	: public AImageFilter< InputImageType, OutputImageType >
{
public:
	typedef AImageFilter< InputImageType, OutputImageType >	PredecessorType;
	typedef typename PredecessorType::Properties		Properties;

	AImageFilterWholeAtOnce( Properties *prop );
protected:

	virtual bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    ) = 0;

	bool
	ExecutionThreadMethod( APipeFilter::UPDATE_TYPE utype );

	
	void
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );
	
	void
	MarkChanges( APipeFilter::UPDATE_TYPE utype );

	void
	AfterComputation( bool successful );

	ReaderBBoxInterface::Ptr	_readerBBox;
	WriterBBoxInterface		*_writerBBox;

private:

};

/**
 * Same usage as template AImageFilterWholeAtOnce, but only when input and output image are the same dimension 
 * and proportions.
 *
 * So only method you must override is ProcessImage().
 **/
template< typename InputImageType, typename OutputImageType >
class AImageFilterWholeAtOnceIExtents
	: public AImageFilterWholeAtOnce< InputImageType, OutputImageType >
{
public:
	typedef AImageFilterWholeAtOnce< InputImageType, OutputImageType >	PredecessorType;
	typedef typename PredecessorType::Properties		Properties;

	AImageFilterWholeAtOnceIExtents( Properties *prop );
protected:

	void
	PrepareOutputDatasets();
private:
	IsSameDimension< ImageTraits< InputImageType >::Dimension, ImageTraits< OutputImageType >::Dimension > ____TestSameDimension; 
};

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/AImageFilterWholeAtOnce.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H*/

/** @} */

