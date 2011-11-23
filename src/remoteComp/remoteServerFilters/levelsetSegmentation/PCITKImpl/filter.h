#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#define CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_

#include "itkThresholdSegmentationLevelSetImageFilter.h"

#include "MedV4D/Common/perfCounter.h"

namespace M4D {
namespace Cell {

template <class TInputImage,class TFeatureImage, class TOutputPixelType = float >
class MySegmtLevelSetFilter	
	: public itk::ThresholdSegmentationLevelSetImageFilter<TInputImage, TFeatureImage, TOutputPixelType>
{
public:
	typedef itk::ThresholdSegmentationLevelSetImageFilter<TInputImage, TFeatureImage, TOutputPixelType> Superclass;	
	typedef MySegmtLevelSetFilter Self;
	typedef itk::SmartPointer<Self> Pointer;
	
	itkNewMacro(Self);
	
	void Initialize(void);	// overriden to measure time spent within
	void PostProcessOutput(); // overriden to stop measuring counter within
	
	void PrintStats(std::ostream &s);
protected:
	MySegmtLevelSetFilter(void);
	~MySegmtLevelSetFilter(void) {}
	
private:
	PerfCounter cntr_;
};
	
//include implementation
#include "src/filter.cxx"

}
}


#endif /*CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_*/
