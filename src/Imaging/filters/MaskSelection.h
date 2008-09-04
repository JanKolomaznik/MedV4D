#ifndef _MASK_SELECTION_H
#define _MASK_SELECTION_H

#include "Common.h"
#include "Imaging/AbstractImageFilter.h"

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MaskSelection.h 
 * @{ 
 **/

namespace Imaging
{

template< typename ImageType >
class MaskSelection
	: public AbstractMultiImageFilter< 2, 1 >
{
public:
	typedef AbstractMultiImageFilter< 2, 1 > 			PredecessorType;
	typedef Image< uint8, ImageTraits<ImageType>::Dimension >	InMaskType;
	typedef ImageTraits< ImageType >::InputPort 			ImageInPort;				
	typedef ImageTraits< ImageType >::OutputPort 			ImageOutPort;				
	typedef typename ImageTraits< InMaskType >::InputPort 		MaskInPort;
	
	class EDifferentMaskExtents
	{
		//TODO
	};


	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};

	MaskSelection( Properties  * prop );
	MaskSelection();
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
	
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/

//include implementation
#include "Imaging/filters/MaskSelection.tcc"

#endif /*_MASK_SELECTION_H*/

