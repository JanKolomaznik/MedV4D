#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_

#include "itkSparseFieldLevelSetImageFilter.h"
//#include "itkThresholdSegmentationLevelSetImageFilter.h"

#include "diffFunc.h"
#include "common/perfCounter.h"

namespace itk
{

template <class TInputImage,class TFeatureImage, class TOutputPixelType = float >
class ThreshSegLevelSetFilter	
	: public itk::SparseFieldLevelSetImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> >
{
public:
	typedef itk::SparseFieldLevelSetImageFilter<TInputImage, Image<TOutputPixelType, TInputImage::ImageDimension> > Superclass;	
	typedef ThreshSegLevelSetFilter Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::ThresholdLevelSetFunc<TInputImage, TFeatureImage> SegmentationFunctionType;
	typedef typename TFeatureImage::PixelType FeaturePixelType;
	
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::ValueType ValueType;
	
	TimeStepType CalculateChange(void);
	void ApplyUpdate(TimeStepType dt);
	
	itkNewMacro(Self);
		
	void GenerateData(void);	// overriden to measure time spent within
	
	void SetUpperThreshold(FeaturePixelType upThreshold) { func_->SetUpperThreshold(upThreshold); }
	void SetLowerThreshold(FeaturePixelType downThreshold) { func_->SetLowerThreshold(downThreshold); }
	void SetPropagationWeight(float32 propWeight) { func_->SetPropagationWeight(propWeight); }
	void SetCurvatureWeight(float32 curvWeight) { func_->SetCurvatureWeight(curvWeight); }
	
	void SetFeatureImage(const TFeatureImage *f)
	  {
	    this->ProcessObject::SetNthInput( 1, const_cast< TFeatureImage * >(f) );
	    func_->SetFeatureImage(f);
	  }
	
	void PrintStats(std::ostream &s);
protected:
	ThreshSegLevelSetFilter(void);
	~ThreshSegLevelSetFilter(void) {}
	
	typename SegmentationFunctionType::Pointer func_;
	
private:
	PerfCounter cntr_;
};

}
//include implementation
#include "src/filter.tcc"

#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
