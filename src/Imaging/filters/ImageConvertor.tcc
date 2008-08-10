#ifndef _IMAGE_CONVERTOR_H
#error File ImageConvertor.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename InputElementType >
ImageConvertor< Image< InputElementType, 3 > >
::ImageConvertor( Properties  * prop )
{
	M4D::Imaging::InputPort *inPort = new InputPortAbstractImage();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO check if OK
	this->_inputPorts.AddPort( inPort );
	this->_outputPorts.AddPort( outPort );
}

template< typename InputElementType >
ImageConvertor< Image< InputElementType, 3 > >
::ImageConvertor()
{
	M4D::Imaging::InputPort *inPort = new InputPortAbstractImage();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO check if OK
	this->_inputPorts.AddPort( inPort );
	this->_outputPorts.AddPort( outPort );
}

template< typename InputElementType >
bool
ImageConvertor< Image< InputElementType, 3 > >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{

}

template< typename InputElementType >
void
ImageConvertor< Image< InputElementType, 3 > >
::PrepareOutputDatasets()
{

}

template< typename InputElementType >
void
ImageConvertor< Image< InputElementType, 3 > >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{

}

template< typename InputElementType >
void
ImageConvertor< Image< InputElementType, 3 > >
::AfterComputation( bool successful )
{

}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_CONVERTOR_H*/
