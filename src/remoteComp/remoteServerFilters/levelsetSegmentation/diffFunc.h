#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "itkThresholdSegmentationLevelSetFunction.h"
#include "common/perfCounter.h"

namespace itk
{

template <class TImageType, class TFeatureImageType = TImageType>
class ThresholdLevelSetFunc
	: public itk::ThresholdSegmentationLevelSetFunction<TImageType, TFeatureImageType>
{
public:
	typedef ThresholdLevelSetFunc<TImageType, TFeatureImageType> Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::ThresholdSegmentationLevelSetFunction<TImageType, TFeatureImageType> Superclass;	
	typedef typename Superclass::PixelType 	PixelType;
	typedef typename Superclass::PixelType 	NeighborhoodType;
	typedef typename Superclass::FloatOffsetType 	FloatOffsetType;
	typedef typename Superclass::RadiusType 	RadiusType;
	
	itkNewMacro(Self);
	
	virtual PixelType ComputeUpdate(
			const NeighborhoodType &neighborhood,
	        void *globalData,
	        const FloatOffsetType& offset = FloatOffsetType(0.0) );
	
	PerfCounter cntr_;
	
protected:
	ThresholdLevelSetFunc();

};

}

//include implementation
#include "src/diffFunc.cxx"

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
