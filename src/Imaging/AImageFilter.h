/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImageFilter.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_H
#define _ABSTRACT_IMAGE_FILTER_H

#include "common/Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AFilter.h"
#include "Imaging/ModificationManager.h"
#include "Imaging/ImageTraits.h"

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
 * Template for abstract image filter with one input and one output.
 **/
template< typename InputImageType, typename OutputImageType >
class AImageFilter: public APipeFilter
{
public:
	typedef APipeFilter	PredecessorType;

	typedef typename ImageTraits< InputImageType >::InputPort
		InputPortType;
	typedef typename ImageTraits< OutputImageType >::OutputPort
		OutputPortType;
	typedef InputImageType	InputImage;
	typedef OutputImageType OutputImage;

	typedef PredecessorType::Properties Properties;

	~AImageFilter() {}
protected:
	AImageFilter( Properties * prop );

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
			int32 		minimums[], 
			int32 		maximums[], 
			float32		elementExtents[]
		    );

	void
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );

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
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AImageFilter );
};

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/AImageFilter.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_H*/

/** @} */

