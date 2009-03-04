#ifndef SERVERLEVELSETSEGMENTATION_H_
#define SERVERLEVELSETSEGMENTATION_H_

#include "itkIntegration/itkFilter.h"
#include "remoteComp/remoteFilterProperties/levelSetRemoteProperties.h"
#include "Imaging/Image.h"

#include "itkImage.h"
//#include "itkThresholdSegmentationLevelSetImageFilter.h"
#include "filter.h"

#include "itkFastMarchingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkZeroCrossingImageFilter.h"
#include "itkCastImageFilter.h"

namespace M4D
{
namespace RemoteComputing
{

template< typename InputElementType, typename OutputElementType >
class ThreshLSSegMedvedWrapper
	: public ITKIntegration::ITKFilter< Imaging::Image<InputElementType, 3>, Imaging::Image<OutputElementType, 3> >
{
public:
	typedef ITKIntegration::ITKFilter< Imaging::Image<InputElementType, 3>, Imaging::Image<OutputElementType, 3> >
		PredecessorType;	
	typedef Imaging::Image<InputElementType, 3> InputImageType;
	typedef Imaging::Image<OutputElementType, 3> OutputImageType;
	typedef LevelSetRemoteProperties<InputElementType, OutputElementType> Properties;
	
	typedef itk::Image<InputElementType, 3> ITKInputImageType;
	typedef itk::Image<OutputElementType, 3> ITKOutputImageType;
	
	ThreshLSSegMedvedWrapper(Properties *props);
	
	void PrepareOutputDatasets(void);
	void ApplyProperties(void);
	
protected:
	bool ProcessImage(
				const InputImageType 	&in,
				OutputImageType		&out
			    );
	
private:
	Properties *properties_;
	
	typedef float32 InternalPixelType;
	typedef itk::Image< InternalPixelType, 3 > InternalITKImageType;
	
	// filter that creates initial level set
	typedef  itk::FastMarchingImageFilter< InternalITKImageType, InternalITKImageType >
	    FastMarchingFilterType;
	
	typedef itk::CastImageFilter< ITKInputImageType, InternalITKImageType > 
			FeatureToFloatFilterType;
		
	// filter that performs actual levelset segmentation
	typedef  itk::ThreshSegLevelSetFilter< 
		InternalITKImageType, InternalITKImageType, InternalPixelType >
			ThresholdSegmentationFilterType;
		
	// threshold filter used to threshold final output to zeros and ones
	typedef itk::BinaryThresholdImageFilter<InternalITKImageType, typename PredecessorType::ITKOutputImageType>
	    ThresholdingFilterType;
	
	typedef itk::CastImageFilter< InternalITKImageType, ITKOutputImageType > 
		FloatToFeatureFilterType;

	
	FastMarchingFilterType::Pointer fastMarching;
	typename ThresholdingFilterType::Pointer thresholder;
	typename ThresholdSegmentationFilterType::Pointer thresholdSegmentation;
	
	typename FeatureToFloatFilterType::Pointer featureToFloatCaster;
	typename FloatToFeatureFilterType::Pointer floatToFeature;
	
	void SetupFastMarchingFilter(void);
	void SetupBinaryThresholder(void);
	void SetupLevelSetSegmentator(void);
	
	void PrintRunInfo(std::ostream &stream);
	
	typedef FastMarchingFilterType::NodeType                NodeType;
	
	NodeType *initSeedNode_;
};

}
}

//include implementation
#include "src/medevedWrapperFilter.cxx"

#endif /*SERVERLEVELSETSEGMENTATION_H_*/
