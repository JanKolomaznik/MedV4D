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
AbstractMultiImageFilter< InCount, OutCount >::AbstractMultiImageFilter( typename AbstractMultiImageFilter< InCount, OutCount >::Properties * prop )
:	AbstractPipeFilter( prop ) 
{

}

template< uint32 InCount, uint32 OutCount >
const AbstractImage&
AbstractMultiImageFilter< InCount, OutCount >::GetInputImage( uint32 idx )const
{
	/*_inputPorts[ idx ].LockDataset();
	return _inputPorts.GetPortTyped< InputPortAbstractImage >( idx ).GetAbstractImage();*/
	return this->GetInputDataSet< AbstractImage >( idx );
}

template< uint32 InCount, uint32 OutCount >
void 
AbstractMultiImageFilter< InCount, OutCount >::ReleaseInputImage( uint32 idx )const
{
	this->ReleaseInputDataSet( idx );
}

template< uint32 InCount, uint32 OutCount >
void
AbstractMultiImageFilter< InCount, OutCount >::ReleaseOutputImage( uint32 idx )const
{
	this->ReleaseOutputDataSet( idx );
}

template< uint32 InCount, uint32 OutCount >
AbstractImage&
AbstractMultiImageFilter< InCount, OutCount >::GetOutputImage( uint32 idx )const
{
	/*_outputPorts[ idx ].LockDataset();
	return _outputPorts.GetPortTyped< OutputPortAbstractImage >( idx ).GetAbstractImage();*/
	return this->GetOutputDataSet< AbstractImage >( idx );
}

template< uint32 InCount, uint32 OutCount >
void
AbstractMultiImageFilter< InCount, OutCount >
::SetOutputImageSize( 
		uint32		idx,
		uint32		dim,
		int32 		minimums[ ], 
		int32 		maximums[ ], 
		float32		elementExtents[ ]
	    )
{
	DIMENSION_TEMPLATE_SWITCH_MACRO( dim,
		ImageFactory::ChangeImageSize( 
				_outputPorts.GetPortTyped< OutputPortTyped<AbstractImage> >( idx ).GetDatasetTyped(),
				Vector< int32, DIM >( minimums ), 
				Vector< int32, DIM >( maximums ), 
				Vector< float32, DIM >( elementExtents )
				);
		);
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
	;/*preparation finished*/
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


