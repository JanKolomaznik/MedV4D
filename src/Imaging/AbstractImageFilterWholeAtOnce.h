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
	: public ImageFilter< InputImageType, OutputImageType >
{
public:
	typedef ImageFilter< InputImageType, OutputImageType >	PredecessorType;
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
	ReaderBBoxInterface::Ptr
	ApplyReaderBBox( const InputImageType &in );

	WriterBBoxInterface *
	ApplyWriterBBox( OutputImageType &out );

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractImageFilterWholeAtOnce.tcc"

#endif /*_ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H*/
