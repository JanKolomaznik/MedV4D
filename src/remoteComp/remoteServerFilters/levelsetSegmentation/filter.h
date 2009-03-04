#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_

#include "itkThresholdSegmentationLevelSetImageFilter.h"

#include "diffFunc.h"
#include "common/perfCounter.h"

namespace itk
{

template <class TInputImage,class TFeatureImage, class TOutputPixelType = float >
class ThreshSegLevelSetFilter	
	: public itk::ThresholdSegmentationLevelSetImageFilter<TInputImage, TFeatureImage, TOutputPixelType>
{
public:
	typedef itk::ThresholdSegmentationLevelSetImageFilter<TInputImage, TFeatureImage, TOutputPixelType> Superclass;	
	typedef ThreshSegLevelSetFilter Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::ThresholdLevelSetFunc<TFeatureImage> SegmentationFunctionType;
	
	typedef typename Superclass::TimeStepType TimeStepType;
	
	TimeStepType CalculateChange(void);
	void ApplyUpdate(TimeStepType dt);
	
	itkNewMacro(Self);
	
	//void Initialize(void);	
	void GenerateData(void);	// overriden to measure time spent within
	
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
#include "src/filter.cxx"

#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
