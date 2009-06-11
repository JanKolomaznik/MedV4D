#ifndef ITKFILTER_H_
#define ITKFILTER_H_

// itk includes
//#include "itkImageWrapper.h"
#include "itkImage.h"

//#include "itkDataContainerWrapper.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "Imaging/ImageFactory.h"
#include "conversions.h"
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

protected:
	typedef typename InputImageType::Element InputPixelType;
	typedef typename OutputImageType::Element OutputPixelType;
	typedef itk::Image< InputPixelType, InputImageType::Dimension >
			ITKInputImageType;
	typedef itk::Image< OutputPixelType, OutputImageType::Dimension >
			ITKOutputImageType;
	
	ITKFilter();
	~ITKFilter() {}	
	
	inline const ITKInputImageType *
		GetInputITKImage(void) { return inITKImage.GetPointer(); }	
	
	inline void
	SetOutputITKImage(const ITKOutputImageType *outImage) { 
		outITKImage = (ITKOutputImageType *) outImage; }
	
	inline ITKOutputImageType *
	GetOutputITKImage(void) { return inITKImage.GetPointer(); }
	
	void PrepareOutputDatasets(void);
	
protected:
	void SetOutImageSize(
			const typename ITKOutputImageType::RegionType &region,
			const typename ITKOutputImageType::SpacingType &spacing);
	
private:
	typename ITKInputImageType::Pointer inITKImage;	
	ITKOutputImageType *outITKImage;
	
	void SetupInITKImageAccordingInMedevedImage(void);
	
	// ITK images that simulate begining and end if ITK pipeline
	//ITKInputImageWrapperType m_inputITKImageWrapper;
};

}}

/** @} */

//include implementation
#include "src/itkFilter.cxx"

#endif /*ITKFILTER_H_*/
