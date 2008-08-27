/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImageFilterWholeAtOnce.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H
#define _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H

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
class AbstractImageFilterWholeAtOnce 
	: public AbstractImageFilter< InputImageType, OutputImageType >
{
public:
	typedef AbstractImageFilter< InputImageType, OutputImageType >	PredecessorType;
	typedef typename PredecessorType::Properties		Properties;

	AbstractImageFilterWholeAtOnce( Properties *prop );
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

};

/**
 * Same usage as template AbstractImageFilterWholeAtOnce, but only when input and output image are the same dimension 
 * and proportions.
 *
 * So only method you must override is ProcessImage().
 **/
template< typename InputImageType, typename OutputImageType >
class AbstractImageFilterWholeAtOnceIExtents
	: public AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >
{
public:
	typedef AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >	PredecessorType;
	typedef typename PredecessorType::Properties		Properties;

	AbstractImageFilterWholeAtOnceIExtents( Properties *prop );
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
#include "Imaging/AbstractImageFilterWholeAtOnce.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H*/

/** @} */

