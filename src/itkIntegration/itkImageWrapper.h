#ifndef ITKCONTAINERWRAPPER_H_
#define ITKCONTAINERWRAPPER_H_

#include "itkImage.h"
#include "Imaging/Image.h"

namespace M4D
{
namespace ITKIntegration
{

/////////////// NOT USED !!!!! Bud left in repo ... ///////////////////

template <typename PixelType, uint16 dimension>
class ITKImageWrapper
	: public itk::Image< PixelType, dimension >
{
public:
	typedef M4D::Imaging::Image< PixelType, dimension > MedvedImageType;
	typedef itk::Image< PixelType, dimension > itkImageType;
	
	ITKImageWrapper();
	
	void SetupAccordingMedvedImage(const MedvedImageType &medImage);
};

}}

//include implementation
#include "src/itkImageWrapper.cxx"

#endif /*ITKIMAGEWRAPPER_H_*/
