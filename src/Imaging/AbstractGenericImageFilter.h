#ifndef _ABSTRACT_GENERIC_IMAGE_FILTER_H
#define _ABSTRACT_GENERIC_IMAGE_FILTER_H

#include "Common.h"
#include "Imaging/AbstractFilter.h"


namespace M4D
{

namespace Imaging
{


template< typename OutputImageType >
class AbstractGenericImageFilter
	: public AbstractPipeFilter
{
public:
	typedef AbstractPipeFilter 		PredecessorType;
	typedef AbstractImagePort		InputPortType;
	typedef ImagePort< OutputImageType > 	OutputPortType;

	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	AbstractGenericImageFilter( Properties  * prop );
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

	/**
	 * Method called in execution methods after computation.
	 * When overriding in successors, predecessor implementation must be called as last.
	 * \param successful Information, whether computation proceeded without problems.
	 **/
	void
	AfterComputation( bool successful );
	
	const AbstractImage	*in;
	Common::TimeStamp	_inTimestamp;
	Common::TimeStamp	_inEditTimestamp;

	OutputImageType		*out;
	Common::TimeStamp	_outTimestamp;
	Common::TimeStamp	_outEditTimestamp;
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractGenericImageFilter.tcc"

#endif /*_ABSTRACT_GENERIC_IMAGE_FILTER_H*/
