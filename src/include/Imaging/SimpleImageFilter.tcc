#ifndef _SIMPLE_IMAGE_FILTER_H
#error File Image.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
ImageFilter< InputImageType, OutputImageType >::ImageFilter()
{
	M4D::Imaging::InputPort *in = new InputPortType();
	M4D::Imaging::OutputPort *out = new OutputPortType();

	//TODO - check whether OK
	_inputPorts.AddPort( in );
	_outputPorts.AddPort( out );
}

template< typename InputImageType, typename OutputImageType >
const InputImageType&
ImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	return _inputPorts.GetPortTyped< InputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
ImageFilter< InputImageType, OutputImageType >::GetOutputImage()const
{
	return _outputPorts.GetPortTyped< OutputPortType >( 0 ).GetImage();
}
//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_SIMPLE_IMAGE_FILTER_H*/

