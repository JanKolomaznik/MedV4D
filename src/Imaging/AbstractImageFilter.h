/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImageFilter.h 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_H
#define _ABSTRACT_IMAGE_FILTER_H

#include "common/Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"
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
class AbstractImageFilter: public AbstractPipeFilter
{
public:
	typedef AbstractPipeFilter	PredecessorType;

	typedef typename ImageTraits< InputImageType >::InputPort
		InputPortType;
	typedef typename ImageTraits< OutputImageType >::OutputPort
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
			int32 		minimums[], 
			int32 		maximums[], 
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

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/AbstractImageFilter.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_H*/

/** @} */

