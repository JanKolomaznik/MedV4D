#ifndef _ABSTRACT_MULTI_IMAGE_FILTER_H
#error File AbstractMultiImageFilter.tcc cannot be included directly!
#else


namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractMultiImageFilter.tcc 
 * @{ 
 **/

namespace Imaging
{


template< uint32 InCount, uint32 OutCount >
AbstractMultiImageFilter< InCount, OutCount >::AbstractImageFilter( typename AbstractImageFilter< InputImageType, OutputImageType >::Properties * prop )
:	AbstractPipeFilter( prop ) 
{

}

const AbstractImage&
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

template< uint32 InCount, uint32 OutCount >
const InputImageType&
AbstractMultiImageFilter< InCount, OutCount >::GetInputImage( uin32 idx )const
{
	_inputPorts.GetPortTyped< InputPortType >( idx ).LockDataset();
	return GetInputImageFromPort( _inputPorts.GetPortTyped< InputPortType >( idx ) );
}

template< uint32 InCount, uint32 OutCount >
void 
AbstractMultiImageFilter< InCount, OutCount >::ReleaseInputImage( uin32 idx )const
{
	_inputPorts.GetPortTyped< InputPortType >( idx ).ReleaseDatasetLock();
}

template< uint32 InCount, uint32 OutCount >
void
AbstractMultiImageFilter< InCount, OutCount >::ReleaseOutputImage( uin32 idx )const
{
	_outputPorts.GetPortTyped< OutputPortType >( idx ).ReleaseDatasetLock();
}

template< uint32 InCount, uint32 OutCount >
OutputImageType&
AbstractMultiImageFilter< InCount, OutCount >::GetOutputImage( uin32 idx )const
{
	_outputPorts.GetPortTyped< OutputPortType >( idx ).LockDataset();
	return GetOutputImageFromPort( _outputPorts.GetPortTyped< OutputPortType >( idx ) );
}

template< uint32 InCount, uint32 OutCount >
void
AbstractMultiImageFilter< InCount, OutCount >
::SetOutputImageSize( 
		uin32		idx,
		int32 		minimums[ ], 
		int32 		maximums[ ], 
		float32		elementExtents[ ]
	    )
{
	_outputPorts.GetPortTyped< OutputPortType >( idx ).SetImageSize( minimums, maximums, elementExtents );
}

template< uint32 InCount, uint32 OutCount >
void
AbstractMultiImageFilter< InCount, OutCount >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );	

	Common::TimeStamp inTS( Common::DefaultTimeStamp );
	Common::TimeStamp outTS( Common::DefaultTimeStamp );
	bool structureChanged = false;

	for( unsigned i = 0; i < InCount; ++i ) {
		in[ i ] = &(this->GetInputImage( i ));
		D_PRINT( "Input Image #" << i << " : " << this->in[ i ] );

		inTS = in[ i ]->GetStructureTimestamp();
		if ( !structureChanged &&
			(!inTS.IdenticalID(_inTimestamp[i]) || inTS != _inTimestamp[i] )
		   )
		{
			structureChanged = true;
		}
	}	

	for( unsigned i = 0; i < OutCount; ++i ) {
		out[ i ] = &(this->GetOutputImage( i ));
		D_PRINT( "Output Image #" << i << " : " << this->out[ i ] );

		outTS = out[ i ]->GetStructureTimestamp();
		if ( !structureChanged &&
			(!outTS.IdenticalID(_outTimestamp[i]) || outTS != _outTimestamp[i] )
		   )
		{
			structureChanged = true;
		}
	}
	if( structureChanged ) {
		utype = AbstractPipeFilter::RECALCULATION;
		this->_callPrepareOutputDatasets = true;
	}

	if( utype == AbstractPipeFilter::ADAPTIVE_CALCULATION ) {
		for( unsigned i=0; i < InCount; ++i ) {
			Common::TimeStamp inEditTS = in[i]->GetModificationManager().GetLastStoredTimestamp();
			if( !inEditTS.IdenticalID( _inEditTimestamp[i] ) || inEditTS > _inEditTimestamp[i] ) 	{
				utype = AbstractPipeFilter::RECALCULATION;
				goto finish;
			}	
		}
		for( unsigned i=0; i < OutCount; ++i ) {
			Common::TimeStamp outEditTS = out[i]->GetModificationManager().GetLastStoredTimestamp();
			if( !outEditTS.IdenticalID( _outEditTimestamp[i] ) || outEditTS != _outEditTimestamp[i] ) 	{
				utype = AbstractPipeFilter::RECALCULATION;
				goto finish;
			}	
		}
	}
finish:
	/*preparation finished*/
}


template< uint32 InCount, uint32 OutCount >
void
AbstractMultiImageFilter< InCount, OutCount >
::AfterComputation( bool successful )
{
	//We store actual timestamps of input and output - for next execution
	for( unsigned i=0; i < InCount; ++i ) {
		_inTimestamp[ i ] = in[ i ]->GetStructureTimestamp();
		_inEditTimestamp[ i ] = in[ i ]->GetEditTimestamp();
		
		this->ReleaseInputImage( i );
	}

	for( unsigned i=0; i < OutCount; ++i ) {
		_outTimestamp[ i ] = out[ i ]->GetStructureTimestamp();
		_outEditTimestamp[ i ] = out[ i ]->GetEditTimestamp();

		this->ReleaseOutputImage( i );
	}


	PredecessorType::AfterComputation( successful );	
}
//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
/** @} */

} /*namespace M4D*/

#endif /*_ABSTRACT_MULTI_IMAGE_FILTER_H*/


