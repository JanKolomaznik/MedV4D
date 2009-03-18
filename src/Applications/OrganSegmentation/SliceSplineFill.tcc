
#ifndef SLICE_SPLINE_FILL_H
#error File SliceSplineFill.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template < typename CoordType >
SliceSplineFill< CoordType >
::SliceSplineFill() : PredecessorType( new Properties() )
{
	this->_name = "SliceSplineFill";

	//TODO check if OK
	InputPortType *inPort = new InputPortType();
	_inputPorts.AppendPort( inPort );

	OutputPortType *outPort = new OutputPortType();
	_outputPorts.AppendPort( outPort );

}

template < typename CoordType >
SliceSplineFill< CoordType >
::SliceSplineFill( typename SliceSplineFill< CoordType >::Properties *prop ) 
: PredecessorType( prop ) 
{
	this->_name = "SliceSplineFill";

	//TODO check if OK
	InputPortType *inPort = new InputPortType();
	_inputPorts.AppendPort( inPort );

	OutputPortType *outPort = new OutputPortType();
	_outputPorts.AppendPort( outPort );
}

template < typename CoordType >
void
SliceSplineFill< CoordType >
::PrepareOutputDatasets()
{
	this->out->UpgradeToExclusiveLock();
		ImageFactory::ChangeImageSize( (*this->out), GetMinimum(), GetMaximum(), GetElementExtents() );
	this->out->DowngradeFromExclusiveLock();
}

template < typename CoordType >
void
SliceSplineFill< CoordType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	utype = AbstractPipeFilter::RECALCULATION;
	this->_callPrepareOutputDatasets = true;

	in = &(this->GetInputDataSet< InputDatasetType >( 0 ));
	out = &(this->GetOutputDataSet< Mask3D >( 0 ));
	
}

template < typename CoordType >
void
SliceSplineFill< CoordType >
::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{
	readerBBox = in->GetWholeDirtyBBox();
	writerBBox = &(out->SetWholeDirtyBBox());
}

template < typename CoordType >
void
SliceSplineFill< CoordType >
::AfterComputation( bool successful )
{
	/*	_inTimestamp[ i ] = in[ i ]->GetStructureTimestamp();
		_inEditTimestamp[ i ] = in[ i ]->GetEditTimestamp();*/
		
	this->ReleaseInputDataSet( 0 );
	this->ReleaseOutputDataSet( 0 );
	PredecessorType::AfterComputation( successful );	
}

template < typename CoordType >
bool
SliceSplineFill< CoordType >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if( !CanContinue() ) return false;

	if ( !(readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		writerBBox->SetState( MS_CANCELED );
		return false;
	}

	for( int32 i = GetMinimum()[2]; i < GetMaximum()[2]; ++i ) {
		if( i < in->GetSliceMin() || i >= in->GetSliceMax() ) {
			ProcessBlankSlice( out->GetSlice( i ) );
		} else {
			ProcessSlice( out->GetSlice( i ), in->GetSlice( i ) );
		}
	}	

	writerBBox->SetModified();
	return true;
}

template < typename CoordType >
void
SliceSplineFill< CoordType >
::ProcessSlice( 
		ImageRegion< uint8, 2 >	slice,
		const typename SliceSplineFill< CoordType >::ObjectsInSlice	&objects 
		)
{
	ProcessBlankSlice( slice );

	for( unsigned i = 0; i < objects.size(); ++i ) {
		Algorithms::FillRegion( slice, objects[i].GetSamplePoints(), InMaskVal );
	}
}

template < typename CoordType >
void
SliceSplineFill< CoordType >
::ProcessBlankSlice( 
		ImageRegion< uint8, 2 >	slice
		)
{
	typename ImageRegion< uint8, 2 >::Iterator iterator = slice.GetIterator();
	while( !iterator.IsEnd() ) {
		*iterator = OutMaskVal;
		++iterator;
	}
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*SLICE_SPLINE_FILL_H*/

