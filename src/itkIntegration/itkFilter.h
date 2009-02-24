#ifndef ITKFILTER_H_
#define ITKFILTER_H_

// itk includes
#include "itkImageWrapper.h"

//#include "itkDataContainerWrapper.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"

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
	typedef ITKImageWrapper< typename InputImageType::Element, InputImageType::Dimension >
		ITKInputImageType;
	typedef ITKImageWrapper< typename OutputImageType::Element, OutputImageType::Dimension >
		ITKOutputImageType;
	
	ITKFilter();
	
	ITKInputImageType &
	GetInputITKImage(void) { return m_inputITKImage; }
	
	void
	SetOutputITKImage(ITKOutputImageType *outImage);
	
	void PrepareOutputDatasets(void);
	
	void SetOutputImageSize(ITKOutputImageType &itkImage);
	
private:
//	ITKDataContainerWrapper< InputImageType::ElementType > m_inputDatCnt;
//	ITKDataContainerWrapper< OutputImageType::ElementType > m_outputDatCnt;
	
	void SetupInputITKImageAccordingInputMedvedImage(void);
	void SetupOutMedvedImageAccordingOutputITKImage(void);
	
	// ITK images that simulate begining and end if ITK pipeline
	ITKInputImageType m_inputITKImage;
	ITKOutputImageType *m_outputITKImage;
};

}}

/** @} */

//include implementation
#include "src/itkFilter.cxx"

#endif /*ITKFILTER_H_*/
