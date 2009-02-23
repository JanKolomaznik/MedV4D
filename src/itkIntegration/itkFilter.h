#ifndef ITKFILTER_H_
#define ITKFILTER_H_

// itk includes
#include "itkImage.h"

//#include "itkDataContainerWrapper.h"
#include "Imaging/AbstractImageFilter.h"

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
	: public Imaging::AbstractImageFilter< InputImageType, InputImageType >
{
public:
	typedef Imaging::AbstractImageFilter< InputImageType, OutputImageType > PredecessorType;
	typedef ITKFilter< InputImageType, OutputImageType > SelfType;

protected:
	typedef itk::Image< typename InputImageType::Element, InputImageType::Dimension >
		ITKInputImageType;
	typedef itk::Image< typename OutputImageType::Element, OutputImageType::Dimension >
		ITKOutputImageType;
	
	ITKFilter();
	
	inline ITKInputImageType &
	GetInputITKImage() { return m_inputITKImage; }
	inline ITKOutputImageType &
	GetOutputITKImage() { return m_outputITKImage; }
	
	void PrepareOutputDatasets(void);
	
	void SetOutputImageSize(ITKOutputImageType &itkImage);
	
private:
//	ITKDataContainerWrapper< InputImageType::ElementType > m_inputDatCnt;
//	ITKDataContainerWrapper< OutputImageType::ElementType > m_outputDatCnt;
	
	// ITK images that simulate begining and end if ITK pipeline
	ITKInputImageType m_inputITKImage;
	ITKOutputImageType m_outputITKImage;
};

}}

/** @} */

//include implementation
#include "src/itkFilter.cxx"

#endif /*ITKFILTER_H_*/
