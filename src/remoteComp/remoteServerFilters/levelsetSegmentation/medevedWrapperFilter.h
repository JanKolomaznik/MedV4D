#ifndef SERVERLEVELSETSEGMENTATION_H_
#define SERVERLEVELSETSEGMENTATION_H_

#include "itkIntegration/itkFilter.h"
#include "remoteComp/iRemoteFilterProperties.h"
#include "Imaging/Image.h"

#include "itkFastMarchingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkZeroCrossingImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImage.h"

#if( 0)//defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
#include "cellSrc/filter.h"
#else
#include "PCSrc/filter.h"
#endif

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
	
	typedef itk::Image<InputElementType, 3> ITKInputImageType;
	typedef itk::Image<OutputElementType, 3> ITKOutputImageType;
	
	struct Properties 
		: public PredecessorType::Properties
		, public iRemoteFilterProperties
	{
		uint32 seedX;
		uint32 seedY;
		uint32 seedZ;
		InputElementType lowerThreshold;
		InputElementType upperThreshold;
		uint32 maxIterations;
		float32 initialDistance;
		float32 curvatureScaling;
		float32 propagationScaling;
		float32 advectionScaling;
		
		Properties()
				: seedX( 64 )
				, seedY( 64 )
				, seedZ( 1 )
				, lowerThreshold( -500 ) 
				, upperThreshold( 500 ) 
				, maxIterations( 800 ) 
				, initialDistance( 5.0f ) 
				, curvatureScaling( 0.01f ) 
				, propagationScaling( 1.0f ) 
				, advectionScaling( 10.0f ) 
			{}
		
		FilterID GetID(void) { return FID_LevelSetSegmentation; }
		
		void SerializeClassInfo(M4D::IO::OutStream &stream) const
		{
			stream.Put<uint16>(FID_LevelSetSegmentation);
			stream.Put<uint16>(GetNumericTypeID< InputElementType >());
			stream.Put<uint16>(GetNumericTypeID< OutputElementType >());
		}
		void SerializeProperties(M4D::IO::OutStream &stream) const
		{
			stream.Put<uint32>(seedX);
			stream.Put<uint32>(seedY);
			stream.Put<uint32>(seedZ);
			stream.Put<InputElementType>(lowerThreshold);
			stream.Put<InputElementType>(upperThreshold);
			stream.Put<uint32>(maxIterations);
			stream.Put<float32>(initialDistance);
			stream.Put<float32>(curvatureScaling);
			stream.Put<float32>(propagationScaling);
			stream.Put<float32>(advectionScaling);
		}
		void DeserializeProperties(M4D::IO::InStream &stream)
		{
			stream.Get<uint32>(seedX);
			stream.Get<uint32>(seedY);
			stream.Get<uint32>(seedZ);
			stream.Get<InputElementType>(lowerThreshold);
			stream.Get<InputElementType>(upperThreshold);
			stream.Get<uint32>(maxIterations);
			stream.Get<float32>(initialDistance);
			stream.Get<float32>(curvatureScaling);
			stream.Get<float32>(propagationScaling);
			stream.Get<float32>(advectionScaling);
		}
	};
	
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
	typedef  itk::MySegmtLevelSetFilter< 
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
#include "medevedWrapperFilter.cxx"

#endif /*SERVERLEVELSETSEGMENTATION_H_*/
