#ifndef _SIMPLE_IMAGE_FILTER_H
#error File Image.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
SimpleImageFilter::Simple2DImageFilter()
{
	InputPort *in = new InputPortType();
	OutputPort *out = new OutputPortType();

	//TODO - check whether OK
	_inputPorts.Add( in );
	_outputPorts.Add( out );
}

template< typename InputImageType, typename OutputImageType >
bool
SimpleImageFilter::ExecutionThreadMethod();
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

template< typename InputImageType, typename OutputImageType >
bool
SimpleImageFilter::ExecutionOnWholeThreadMethod()
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

template< typename InputImageType, typename OutputImageType >
const InputImageType&
SimpleImageFilter::GetInputImage()const
{
	return _inputPorts.GetPortTyped< InputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
SimpleImageFilter::GetOutputImage()const
{
	return _outputPorts.GetPortTyped< OutputPortType >( 0 ).GetImage();
}
//******************************************************************************
//******************************************************************************

/*
template< typename InputElementType, typename OutputElementType >
Simple2DImageFilter::Simple2DImageFilter()
{
	InputPort *in = new InputPortImageFilter< Image2D< InputElementType > >();
	OutputPort *out = new OutputPortImageFilter< Image2D< OutputElementType > >();

	//TODO
	_inputPorts.Add( in );
	_outputPorts.Add( out );
}

template< typename InputElementType, typename OutputElementType >
bool
Simple2DImageFilter::ExecutionThreadMethod();
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

template< typename InputElementType, typename OutputElementType >
bool
Simple2DImageFilter::ExecutionOnWholeThreadMethod()
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

//******************************************************************************

template< typename InputElementType, typename OutputElementType >
Simple3DImageFilter::Simple3DImageFilter()
{
	InputPort *in = NULL;
	OutputPort *out = NULL;

	//TODO
	_inputPorts.Add( in );
	_outputPorts.Add( out );
}

template< typename InputElementType, typename OutputElementType >
bool
Simple2DImageFilter::ExecutionThreadMethod();
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

template< typename InputElementType, typename OutputElementType >
bool
Simple2DImageFilter::ExecutionOnWholeThreadMethod()
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

//******************************************************************************

template< typename InputElementType, typename OutputElementType >
SimpleSlicedImageFilter::SimpleSlicedImageFilter()
{
	//TODO
}


template< typename InputElementType, typename OutputElementType >
bool
Simple2DImageFilter::ExecutionThreadMethod();
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

template< typename InputElementType, typename OutputElementType >
bool
Simple2DImageFilter::ExecutionOnWholeThreadMethod()
{
	//TODO
	
	if( !CanContinue() ) {
		return false;
	}
	return true;
}

*/


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_SIMPLE_IMAGE_FILTER_H*/

