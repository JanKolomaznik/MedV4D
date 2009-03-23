/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageConvertor.h 
 * @{ 
 **/

#ifndef _IMAGE_CONVERTOR_H
#define _IMAGE_CONVERTOR_H

#include "common/Common.h"
#include "Imaging/AbstractImageFilter.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

class DefaultConvertor
{
public:

	template< typename InputType, typename OutputType >
	static void
	Convert( const InputType &input, OutputType &output )
		{
			output = static_cast<OutputType>(input);
		}
};

template< typename OutputImageType, typename Convertor = DefaultConvertor >
class ImageConvertor
	: public AbstractImageFilter< AbstractImage, OutputImageType >
{
public:
	typedef AbstractImageFilter< AbstractImage, OutputImageType > 	PredecessorType;

	class EDatasetConversionImpossible
	{
		//TODO
	};

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	ImageConvertor( Properties  * prop );
	ImageConvertor();
protected:
	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );

	/**
	 * Method which will prepare datasets connected by output ports.
	 * Set their extents, etc.
	 **/
	void
	PrepareOutputDatasets();

	/**
	 * Method called in execution methods before actual computation.
	 * When overriding in successors predecessor implementation must be called first.
	 * \param utype Input/output parameter choosing desired update method. If 
	 * desired update method can't be used - right type is put as output value.
	 **/
	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype );

	/**
	 * Method called in execution methods after computation.
	 * When overriding in successors, predecessor implementation must be called as last.
	 * \param successful Information, whether computation proceeded without problems.
	 **/
	void
	AfterComputation( bool successful );
	
	ReaderBBoxInterface::Ptr	_readerBBox;
	WriterBBoxInterface		*_writerBBox;

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ImageConvertor.tcc"

#endif /*_IMAGE_CONVERTOR_H*/

/** @} */

