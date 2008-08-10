#ifndef _ABSTRACT_GENERIC_IMAGE_FILTER_H
#error File AbstractGenericImageFilter.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename OutputImageType >
AbstractGenericImageFilter
::AbstractGenericImageFilter( Properties  * prop )
	: PredecessorType( prop ), 
	in( NULL ), _inTimestamp( Common::DefaultTimeStamp ), _inEditTimestamp( Common::DefaultTimeStamp ), 
	out( NULL ), _outTimestamp( Common::DefaultTimeStamp ), _outEditTimestamp( Common::DefaultTimeStamp )
{
	M4D::Imaging::InputPort *inPort = new InputPortType();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO check if OK
	this->_inputPorts.AddPort( inPort );
	this->_outputPorts.AddPort( outPort );
}

template< typename OutputImageType >
void
AbstractGenericImageFilter
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	_inputPorts.GetPortTyped< InputPortType >( 0 ).LockDataset();
	this->in = &( _inputPorts.GetPortTyped< InputPortType >( 0 ).GetAbstractImage() );

	_outputPorts.GetPortTyped< OutputPortType >( 0 ).LockDataset();
	this->out = &(_outputPorts.GetPortTyped< OutputPortType >( 0 ).GetImage());
	
	//timestamp check
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
		_inTimestamp = inTS;
		_outTimestamp = outTS;
		PrepareOutputDatasets();
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

template< typename OutputImageType >
void
AbstractGenericImageFilter
::AfterComputation( bool successful )
{
	//We store actual timestamps of input and output - for next execution
	_inEditTimestamp = in->GetModificationManager().GetActualTimestamp();
	_outEditTimestamp = out->GetModificationManager().GetActualTimestamp();

	in = NULL;
	_inputPorts.GetPortTyped< InputPortType >( 0 ).ReleaseDatasetLock();
	out = NULL;
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).ReleaseDatasetLock();

	PredecessorType::AfterComputation( successful );
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_GENERIC_IMAGE_FILTER_H*/
