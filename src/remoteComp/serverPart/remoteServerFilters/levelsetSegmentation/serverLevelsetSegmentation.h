#ifndef SERVERLEVELSETSEGMENTATION_H_
#define SERVERLEVELSETSEGMENTATION_H_

#include "itkIntegration/itkFilter.h"
#include "remoteComp/remoteFilterProperties/levelSetRemoteProperties.h"
#include "Imaging/Image.h"

#include "itkImage.h"
#include "itkThresholdSegmentationLevelSetImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkZeroCrossingImageFilter.h"

namespace M4D
{
namespace RemoteComputing
{

template< typename InputElementType, typename OutputElementType >
class ServerLevelsetSegmentation
	: public ITKIntegration::ITKFilter< Imaging::Image<InputElementType, 3>, Imaging::Image<OutputElementType, 3> >
{
public:
	typedef ITKIntegration::ITKFilter< Imaging::Image<InputElementType, 3>, Imaging::Image<OutputElementType, 3> >
		PredecessorType;	
	typedef Imaging::Image<InputElementType, 3> InputImageType;
	typedef Imaging::Image<OutputElementType, 3> OutputImageType;
	typedef LevelSetRemoteProperties<InputElementType, OutputElementType> Properties;
	
	ServerLevelsetSegmentation();
	
	void PrepareOutputDatasets(void);
	
	inline Properties * GetProperties(void) { return &properties_; }
	void ApplyProperties(void);
	
protected:
	bool ProcessImage(
				const InputImageType 	&in,
				OutputImageType		&out
			    );
	
private:
	Properties properties_;
	
	typedef float32 InternalPixelType;
	typedef itk::Image< InternalPixelType, 3 > InternalITKImageType;
	
	// filter that creates initial level set
	typedef  itk::FastMarchingImageFilter< InternalITKImageType, InternalITKImageType >
	    FastMarchingFilterType;
		
	// filter that performs actual levelset segmentation
	typedef  itk::ThresholdSegmentationLevelSetImageFilter< 
		InternalITKImageType, typename PredecessorType::ITKInputImageType, InternalPixelType >
			ThresholdSegmentationLevelSetImageFilterType;
		
	// threshold filter used to threshold final output to zeros and ones
	typedef itk::BinaryThresholdImageFilter<InternalITKImageType, typename PredecessorType::ITKOutputImageType>
	    ThresholdingFilterType;

	
	FastMarchingFilterType::Pointer fastMarching;
	typename ThresholdingFilterType::Pointer thresholder;
	typename ThresholdSegmentationLevelSetImageFilterType::Pointer thresholdSegmentation;
	
	void SetupFastMarchingFilter(void);
	void SetupBinaryThresholder(void);
	void SetupLevelSetSegmentator(void);
	
	typedef FastMarchingFilterType::NodeType                NodeType;
	
	NodeType *initSeedNode_;
};

}
}

//include implementation
#include "src/serverLevelsetSegmentation.cxx"

#endif /*SERVERLEVELSETSEGMENTATION_H_*/
