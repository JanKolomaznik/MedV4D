#ifndef _BASIC_IMAGE_FILTERS_H
#error File BasicImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
CopyImageFilter< InputImageType, OutputImageType >::CopyImageFilter()
{

}

template< typename InputImageType, typename OutputImageType >
bool
CopyImageFilter< InputImageType, OutputImageType >::ExecutionThreadMethod()
{
	//TODO
	const InputImageType &inImage = this->GetInputImage();
	OutputImageType &outImage = this->GetOutputImage();

	if( !this->CanContinue() ) {
		return false;
	}
	return true;
}

template< typename InputImageType, typename OutputImageType >
bool
CopyImageFilter< InputImageType, OutputImageType >::ExecutionOnWholeThreadMethod()
{
	//TODO
	const InputImageType &inImage = this->GetInputImage();
	OutputImageType &outImage = this->GetOutputImage();
	
	if( !this->CanContinue() ) {
		return false;
	}
	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_BASIC_IMAGE_FILTERS_H*/
