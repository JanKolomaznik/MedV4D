#ifndef _MASK_SELECTION_H
#define _MASK_SELECTION_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/AMultiImageFilter.h"

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

template< uint32 Dim >
struct MaskSelectionHelper
{

	ReaderBBoxInterface::Ptr	_imageReaderBBox;
	ReaderBBoxInterface::Ptr	_maskReaderBBox;
	WriterBBoxInterface		*_writerBBox;
};

template< typename ImageType >
class MaskSelection
	: public AMultiImageFilter< 2, 1 >
{
public:
	typedef AMultiImageFilter< 2, 1 > 				PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType 		ElementType;
	typedef Image< uint8, ImageTraits<ImageType>::Dimension >	InMaskType;
	typedef typename ImageTraits< ImageType >::InputPort 		ImageInPort;
	typedef typename ImageTraits< ImageType >::OutputPort 		ImageOutPort;
	typedef typename ImageTraits< InMaskType >::InputPort 		MaskInPort;
	
	class EDifferentMaskExtents
	{
		//TODO
	};


	struct Properties : public PredecessorType::Properties
	{
		Properties(): background( 0 ) {}

		ElementType	background;
	};

	MaskSelection( Properties  * prop );
	MaskSelection();


	GET_SET_PROPERTY_METHOD_MACRO( ElementType, Background, background );
protected:
	bool
	ExecutionThreadMethod( APipeFilter::UPDATE_TYPE utype );

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
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );

	void
	MarkChanges( APipeFilter::UPDATE_TYPE utype );

	/**
	 * Method called in execution methods after computation.
	 * When overriding in successors, predecessor implementation must be called as last.
	 * \param successful Information, whether computation proceeded without problems.
	 **/
	void
	AfterComputation( bool successful );

	MaskSelectionHelper< ImageTraits< ImageType >::Dimension > _helper;
	
private:
	template< uint32 Dim >
	void
	MarkChangesHelper( APipeFilter::UPDATE_TYPE utype );

	template< uint32 Dim >
	bool
	ExecutionThreadMethodHelper( APipeFilter::UPDATE_TYPE utype );

	bool
	Process();

	GET_PROPERTIES_DEFINITION_MACRO;

};

	
} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/

//include implementation
#include "MedV4D/Imaging/filters/MaskSelection.tcc"

#endif /*_MASK_SELECTION_H*/

