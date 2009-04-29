#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_

#ifdef WITH_ORIG_ITKPIPELINE
#include "itkThresholdSegmentationLevelSetImageFilter.h"
#else
#include "itkSparseFieldLevelSetImageFilter.h"
#endif


#include "diffFunc.h"
#include "common/perfCounter.h"

namespace itk
{

template <class TInputImage,class TFeatureImage, class TOutputPixelType = float >
class MySegmtLevelSetFilter	
#ifdef WITH_ORIG_ITKPIPELINE
	: public itk::ThresholdSegmentationLevelSetImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> >
#else
	: public itk::SparseFieldLevelSetImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> >
#endif
{
public:
#ifdef WITH_ORIG_ITKPIPELINE
	typedef itk::ThresholdSegmentationLevelSetImageFilter<TInputImage, TFeatureImage, TOutputPixelType > Superclass;
#else
	typedef itk::SparseFieldLevelSetImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;
#endif
	typedef MySegmtLevelSetFilter Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::ThresholdLevelSetFunc<TInputImage, TFeatureImage> SegmentationFunctionType;
	typedef typename TFeatureImage::PixelType FeaturePixelType;
	
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::ValueType ValueType;
	
	TimeStepType CalculateChange(void);
	void ApplyUpdate(TimeStepType dt);
	
	itkNewMacro(Self);
		
	void GenerateData(void);	// overriden to measure time spent within
	
#ifndef WITH_ORIG_ITKPIPELINE
	void SetUpperThreshold(FeaturePixelType upThreshold) { func_->SetUpperThreshold(upThreshold); }
	void SetLowerThreshold(FeaturePixelType downThreshold) { func_->SetLowerThreshold(downThreshold); }
	void SetPropagationScaling(float32 propWeight) { func_->SetPropagationWeight(propWeight); }
	void SetCurvatureScaling(float32 curvWeight) { func_->SetCurvatureWeight(curvWeight); }
	
	void SetFeatureImage(const TFeatureImage *f)
	  {
	    this->ProcessObject::SetNthInput( 1, const_cast< TFeatureImage * >(f) );
	    func_->SetFeatureImage(f);
	  }
#endif
	
	void PrintStats(std::ostream &s);
protected:
	MySegmtLevelSetFilter(void);
	~MySegmtLevelSetFilter(void) {}
	
	typename SegmentationFunctionType::Pointer func_;
	
private:
	PerfCounter cntr_;
};

}
//include implementation
#include "src/filter.tcc"

#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
