#ifndef ITKFILTER_H_
#define ITKFILTER_H_

// itk includes
//#include "itkImageWrapper.h"
#include "itkImage.h"

//#include "itkDataContainerWrapper.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "Imaging/ImageFactory.h"

/**
 *  @addtogroup itkIntegration ITK Integration
 *  @{
 */

namespace M4D
{
namespace ITKIntegration
{

// currently only for image types ...
template< typename InputImageType, typename OutputImageType >
class ITKFilter 
	: public Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >
{
public:
	typedef M4D::Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;
	typedef ITKFilter< InputImageType, OutputImageType > SelfType;
//	typedef typename InputImageType InputImageType;	
//	typedef typename OutputImageType OutputImageType;

protected:
	typedef typename InputImageType::Element InputPixelType;
	typedef typename OutputImageType::Element OutputPixelType;
//	typedef ITKImageWrapper< InputPixelType, InputImageType::Dimension >
//		ITKInputImageWrapperType;
	typedef itk::Image< InputPixelType, InputImageType::Dimension >
			ITKInputImageType;
//	typedef ITKImageWrapper< OutputPixelType, OutputImageType::Dimension >
//		ITKOutputImageWrapperType;
	typedef itk::Image< OutputPixelType, OutputImageType::Dimension >
			ITKOutputImageType;
	
	ITKFilter();
	~ITKFilter() {}
	
	const ITKInputImageType *
	GetInputITKImage(void) { return inITKImage.GetPointer(); }	
	void
	SetOutputITKImage(const ITKOutputImageType *outImage) { 
		outITKImage = (ITKOutputImageType *) outImage; }
	
	void PrepareOutputDatasets(void);	
	void SetOutputImageSizeAccordingITK(ITKOutputImageType *itkImage);
	
private:
	typename ITKInputImageType::Pointer inITKImage;	
	ITKOutputImageType *outITKImage;
	
	void SetupOutMedvedImageAccordingOutputITKImage(void);
	void SetupInITKImageAccordingInMedevedImage(void);
	
	// ITK images that simulate begining and end if ITK pipeline
	//ITKInputImageWrapperType m_inputITKImageWrapper;
};

}}

/** @} */

//include implementation
#include "src/itkFilter.cxx"

#endif /*ITKFILTER_H_*/
