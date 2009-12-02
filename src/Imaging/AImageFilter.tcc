/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImageFilter.tcc 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_H
#error File AImageFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
AImageFilter< InputImageType, OutputImageType >::AImageFilter( typename AImageFilter< InputImageType, OutputImageType >::Properties * prop )
:	APipeFilter( prop ), 
	in( NULL ), _inTimestamp( Common::DefaultTimeStamp ), _inEditTimestamp( Common::DefaultTimeStamp ), 
	out( NULL ), _outTimestamp( Common::DefaultTimeStamp ), _outEditTimestamp( Common::DefaultTimeStamp )
{
	M4D::Imaging::InputPort *inPort = new InputPortType();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO check if OK
	_inputPorts.AppendPort( inPort );
	_outputPorts.AppendPort( outPort );
}

/*const AImage&
GetInputImageFromPort( InputPortAImage &port ); //definition in .cpp

template< typename ImageType >
const ImageType&
GetInputImageFromPort( InputPortImageFilter< ImageType > &port )
{
	return port.GetImage();
}	

AImage&
GetOutputImageFromPort( OutputPortAImage &port ); //definition in .cpp

template< typename ImageType >
ImageType&
GetOutputImageFromPort( OutputPortImageFilter< ImageType > &port )
{
	return port.GetImage();
}	
*/
template< typename InputImageType, typename OutputImageType >
const InputImageType&
AImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	/*_inputPorts.GetPortTyped< InputPortType >( 0 ).LockDataset();
	return GetInputImageFromPort( _inputPorts.GetPortTyped< InputPortType >( 0 ) );*/
	return this->GetInputDataset< InputImageType >( 0 );
}

template< typename InputImageType, typename OutputImageType >
void 
AImageFilter< InputImageType, OutputImageType >::ReleaseInputImage()const
{
	this->ReleaseInputDataset( 0 );
	//_inputPorts.GetPortTyped< InputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilter< InputImageType, OutputImageType >::ReleaseOutputImage()const
{
	//_outputPorts.GetPortTyped< OutputPortType >( 0 ).ReleaseDatasetLock();
	this->ReleaseOutputDataset( 0 );
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
AImageFilter< InputImageType, OutputImageType >::GetOutputImage()const
{
	/*_outputPorts.GetPortTyped< OutputPortType >( 0 ).LockDataset();
	return GetOutputImageFromPort( _outputPorts.GetPortTyped< OutputPortType >( 0 ) );*/
	return this->GetOutputDataset< OutputImageType >( 0 );
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilter< InputImageType, OutputImageType >
::SetOutputImageSize( 
		int32 		minimums[ ], 
		int32 		maximums[ ], 
		float32		elementExtents[ ]
	    )
{
	this->out->UpgradeToExclusiveLock();
	ImageFactory::ChangeImageSize( 
			*(this->out),
			//_outputPorts.GetPortTyped< OutputPortTyped<OutputImageType> >( 0 ).GetDatasetTyped(),
			Vector< int32, OutputImageType::Dimension >( minimums ), 
			Vector< int32, OutputImageType::Dimension >( maximums ), 
			Vector< float32, OutputImageType::Dimension >( elementExtents )
			);
	this->out->DowngradeFromExclusiveLock();
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilter< InputImageType, OutputImageType >
::BeforeComputation( APipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );	
	
	this->in = NULL;
	this->out = NULL;
	this->in = &(this->GetInputImage());
	this->out = &(this->GetOutputImage());

	D_PRINT( "Input Image : " << this->in );
	D_PRINT( "Output Image : " << this->out );

	Common::TimeStamp inTS = in->GetStructureTimestamp();
	Common::TimeStamp outTS = out->GetStructureTimestamp();

	//Check whether structure of images changed
	if ( 
		!inTS.IdenticalID( _inTimestamp )
		|| inTS != _inTimestamp
		|| !outTS.IdenticalID( _outTimestamp )
		|| outTS != _outTimestamp 
	) {
		utype = APipeFilter::RECALCULATION;
		this->_callPrepareOutputDatasets = true;
	}
	if( utype == APipeFilter::ADAPTIVE_CALCULATION ) {
		Common::TimeStamp inEditTS = in->GetModificationManager().GetLastStoredTimestamp();
		Common::TimeStamp outEditTS = out->GetModificationManager().GetActualTimestamp();
		if( 
			!inEditTS.IdenticalID( _inEditTimestamp ) 
			|| !outEditTS.IdenticalID( _outEditTimestamp )
			|| inEditTS > _inEditTimestamp 
			|| outEditTS != _outEditTimestamp
		) {
			utype = APipeFilter::RECALCULATION;
		}
	}
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilter< InputImageType, OutputImageType >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

}


template< typename InputImageType, typename OutputImageType >
void
AImageFilter< InputImageType, OutputImageType >
::AfterComputation( bool successful )
{
	//We store actual timestamps of input and output - for next execution
	if( this->in ) {
		_inEditTimestamp = in->GetModificationManager().GetActualTimestamp();
		_inTimestamp = in->GetStructureTimestamp();
		this->ReleaseInputImage();
	}

	if( this->out ) {
		_outEditTimestamp = out->GetModificationManager().GetActualTimestamp();
		_outTimestamp = out->GetStructureTimestamp();
		this->ReleaseOutputImage();
	}

	PredecessorType::AfterComputation( successful );	
}
//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_IMAGE_FILTER_H*/


/** @} */

