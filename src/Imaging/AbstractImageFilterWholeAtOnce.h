#ifndef _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H
#define _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H

#include "Common.h"
#include "Imaging/AbstractImageFilter.h"
#include <vector>

namespace M4D
{
namespace Imaging
{


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

//include implementation
#include "Imaging/AbstractImageFilterWholeAtOnce.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H*/
