/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractImageFilter.tcc 
 * @{ 
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_H
#error File AbstractImageFilter.tcc cannot be included directly!
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
AbstractImageFilter< InputImageType, OutputImageType >::AbstractImageFilter( typename AbstractImageFilter< InputImageType, OutputImageType >::Properties * prop )
:	AbstractPipeFilter( prop ), 
	in( NULL ), _inTimestamp( Common::DefaultTimeStamp ), _inEditTimestamp( Common::DefaultTimeStamp ), 
	out( NULL ), _outTimestamp( Common::DefaultTimeStamp ), _outEditTimestamp( Common::DefaultTimeStamp )
{
	M4D::Imaging::InputPort *inPort = new InputPortType();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO check if OK
	_inputPorts.AddPort( inPort );
	_outputPorts.AddPort( outPort );
}

/*const AbstractImage&
GetInputImageFromPort( InputPortAbstractImage &port ); //definition in .cpp

template< typename ImageType >
const ImageType&
GetInputImageFromPort( InputPortImageFilter< ImageType > &port )
{
	return port.GetImage();
}	

AbstractImage&
GetOutputImageFromPort( OutputPortAbstractImage &port ); //definition in .cpp

template< typename ImageType >
ImageType&
GetOutputImageFromPort( OutputPortImageFilter< ImageType > &port )
{
	return port.GetImage();
}	
*/
template< typename InputImageType, typename OutputImageType >
const InputImageType&
AbstractImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	/*_inputPorts.GetPortTyped< InputPortType >( 0 ).LockDataset();
	return GetInputImageFromPort( _inputPorts.GetPortTyped< InputPortType >( 0 ) );*/
	return this->GetInputDataSet< InputImageType >( 0 );
}

template< typename InputImageType, typename OutputImageType >
void 
AbstractImageFilter< InputImageType, OutputImageType >::ReleaseInputImage()const
{
	this->ReleaseInputDataSet( 0 );
	//_inputPorts.GetPortTyped< InputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >::ReleaseOutputImage()const
{
	//_outputPorts.GetPortTyped< OutputPortType >( 0 ).ReleaseDatasetLock();
	this->ReleaseOutputDataSet( 0 );
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
AbstractImageFilter< InputImageType, OutputImageType >::GetOutputImage()const
{
	/*_outputPorts.GetPortTyped< OutputPortType >( 0 ).LockDataset();
	return GetOutputImageFromPort( _outputPorts.GetPortTyped< OutputPortType >( 0 ) );*/
	return this->GetOutputDataSet< OutputImageType >( 0 );
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
::SetOutputImageSize( 
		int32 		minimums[ ], 
		int32 		maximums[ ], 
		float32		elementExtents[ ]
	    )
{
	ImageFactory::ChangeImageSize( 
			_outputPorts.GetPortTyped< OutputPortTyped<OutputImageType> >( 0 ).GetDatasetTyped(),
			OutputImageType::Dimension, 
			minimums, 
			maximums, 
			elementExtents
			);
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );	
	
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
		utype = AbstractPipeFilter::RECALCULATION;
		this->_callPrepareOutputDatasets = true;
	}
	if( utype == AbstractPipeFilter::ADAPTIVE_CALCULATION ) {
		Common::TimeStamp inEditTS = in->GetModificationManager().GetLastStoredTimestamp();
		Common::TimeStamp outEditTS = out->GetModificationManager().GetActualTimestamp();
		if( 
			!inEditTS.IdenticalID( _inEditTimestamp ) 
			|| !outEditTS.IdenticalID( _outEditTimestamp )
			|| inEditTS > _inEditTimestamp 
			|| outEditTS != _outEditTimestamp
		) {
			utype = AbstractPipeFilter::RECALCULATION;
		}
	}
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

}


template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
::AfterComputation( bool successful )
{
	//We store actual timestamps of input and output - for next execution
	_inEditTimestamp = in->GetModificationManager().GetActualTimestamp();
	_outEditTimestamp = out->GetModificationManager().GetActualTimestamp();
	_inTimestamp = in->GetStructureTimestamp();
	_outTimestamp = out->GetStructureTimestamp();

	this->ReleaseInputImage();
	this->ReleaseOutputImage();

	PredecessorType::AfterComputation( successful );	
}
//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_IMAGE_FILTER_H*/


/** @} */

