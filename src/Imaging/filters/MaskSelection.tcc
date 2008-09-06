#ifndef _MASK_SELECTION_H
#error File MaskSelection.tcc cannot be included directly!
#else

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MaskSelection.tcc 
 * @{ 
 **/

namespace Imaging
{


//******************************************************************************

template< typename ImageType >
MaskSelection< ImageType >
::MaskSelection( typename MaskSelection< ImageType >::Properties  * prop )
	: PredecessorType( prop )
{
	M4D::Imaging::InputPort *imageInport = new ImageInPort();
	M4D::Imaging::InputPort *maskInPort = new MaskInPort();

	M4D::Imaging::OutputPort *imageOutport = new ImageOutPort();

	this->_inputPorts.AddPort( imageInport );
	this->_inputPorts.AddPort( maskInPort );
	this->_outputPorts.AddPort( imageOutport );
}

template< typename ImageType >
MaskSelection< ImageType >
::MaskSelection()
	: PredecessorType( new Properties() )
{
	M4D::Imaging::InputPort *imageInport = new ImageInPort();
	M4D::Imaging::InputPort *maskInPort = new MaskInPort();

	M4D::Imaging::OutputPort *imageOutport = new ImageOutPort();

	this->_inputPorts.AddPort( imageInport );
	this->_inputPorts.AddPort( maskInPort );
	this->_outputPorts.AddPort( imageOutport );
}

template< typename ImageType >
bool
MaskSelection< ImageType >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	return ExecutionThreadMethodHelper< ImageTraits< ImageType >::Dimension >( utype );

}

template< typename ImageType >
void
MaskSelection< ImageType >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	int32 minimums[ ImageTraits<ImageType>::Dimension ];
	int32 maximums[ ImageTraits<ImageType>::Dimension ];
	float32 voxelExtents[ ImageTraits<ImageType>::Dimension ];

	for( unsigned i=0; i < ImageTraits<ImageType>::Dimension; ++i ) {
		const DimensionExtents & dimExt = this->in[0]->GetDimensionExtents( i );
		const DimensionExtents & maskDimExt = this->in[1]->GetDimensionExtents( i );
		
		if( dimExt != maskDimExt ) {
			throw EDifferentMaskExtents();
		}

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent;
	}
	

	this->SetOutputImageSize( 0, ImageTraits<ImageType>::Dimension, minimums, maximums, voxelExtents );
}

template< typename ImageType >
void
MaskSelection< ImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );
}

template< typename ImageType >
void
MaskSelection< ImageType >
::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{
	MarkChangesHelper< ImageTraits< ImageType >::Dimension >( utype );
}

template< typename ImageType >
void
MaskSelection< ImageType >
::AfterComputation( bool successful )
{
	PredecessorType::AfterComputation( successful );
}

template< typename ImageType >
template< uint32 Dim >
void
MaskSelection< ImageType >
::MarkChangesHelper( AbstractPipeFilter::UPDATE_TYPE utype )
{

}

template< typename ImageType >
template< uint32 Dim >
bool
MaskSelection< ImageType >
::ExecutionThreadMethodHelper( AbstractPipeFilter::UPDATE_TYPE utype )
{
	/*utype = utype;
	D_BLOCK_COMMENT( "++++ Entering ExecutionThreadMethod() - ImageConvertor", "----- Leaving MainExecutionThread() - ImageConvertor" );
	if ( !( _readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		_writerBBox->SetState( MS_CANCELED );
		return false;
	}
	bool result = false;
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
			this->in->GetElementTypeID(), 
			result = ConvertImage< TTYPE, typename ImageTraits< OutputImageType >::ElementType, Convertor >( *(this->in), *(this->out) )
			);
	if( result ) {
		_writerBBox->SetModified();
	} else {
		_writerBBox->SetState( MS_CANCELED );
	}
	return result;*/
	return false;
}

} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/


#endif /*_MASK_SELECTION_H*/

