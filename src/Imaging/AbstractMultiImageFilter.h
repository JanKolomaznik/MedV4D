#ifndef _ABSTRACT_MULTI_IMAGE_FILTER_H
#define _ABSTRACT_MULTI_IMAGE_FILTER_H

#include "common/Common.h"

#include "Imaging/Ports.h"
#include "Imaging/ImageDataTemplate.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/ModificationManager.h"
#include "Imaging/ImageTraits.h"

#include <vector>

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractMultiImageFilter.h 
 * @{ 
 **/

namespace Imaging
{

template< uint32 InCount, uint32 OutCount >
class AbstractMultiImageFilter: public AbstractPipeFilter
{
public:
	typedef AbstractPipeFilter	PredecessorType;

	struct Properties : public PredecessorType::Properties
	{

	};

	~AbstractMultiImageFilter() {}
protected:
	AbstractMultiImageFilter( Properties * prop );

	const AbstractImage&
	GetInputImage( uint32 idx )const;

	void
	ReleaseInputImage( uint32 idx )const;

	AbstractImage&
	GetOutputImage( uint32 idx )const;

	void
	ReleaseOutputImage( uint32 idx )const;

	void
	SetOutputImageSize(
			uint32		idx,
			uint32		dim,
			int32 		minimums[], 
			int32 		maximums[], 
			float32		elementExtents[]
		    );

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	AfterComputation( bool successful );


	const AbstractImage	*in[ InCount ];
	Common::TimeStamp	_inTimestamp[ InCount ];
	Common::TimeStamp	_inEditTimestamp[ InCount ];


	AbstractImage		*out[ OutCount ];
	Common::TimeStamp	_outTimestamp[ OutCount ];
	Common::TimeStamp	_outEditTimestamp[ OutCount ];

	
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractMultiImageFilter );
};

} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractMultiImageFilter.tcc"

#endif /*_ABSTRACT_MULTI_IMAGE_FILTER_H*/

