#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "itkFiniteDifferenceFunction.h"
#include "common/perfCounter.h"
#include "speedTermSolver.h"
#include "advectionTermSolver.h"
#include "curvatureTermSolver.h"

namespace itk
{

template <class TImageType, class TFeatureImageType = TImageType>
class ThresholdLevelSetFunc
	: public itk::FiniteDifferenceFunction<TImageType>
	, public SpeedTermSolver
	, public AdvectionTermSolver,
	, public CurvatureTermSolver
{
public:
	typedef ThresholdLevelSetFunc<TImageType, TFeatureImageType> Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::FiniteDifferenceFunction<TImageType> Superclass;	
	typedef typename Superclass::PixelType 	PixelType;
	typedef typename Superclass::PixelType 	NeighborhoodType;
	typedef typename Superclass::FloatOffsetType 	FloatOffsetType;
	typedef typename Superclass::RadiusType 	RadiusType;
	typedef typename Superclass::ScalarValueType ScalarValueType;
	typedef typename Superclass::TimeStepType TimeStepType;
	
	itkNewMacro(Self);
	
	virtual PixelType ComputeUpdate(
			const NeighborhoodType &neighborhood,
	        void *globalData,
	        const FloatOffsetType& offset = FloatOffsetType(0.0) );
	
	TimeStepType ComputeGlobalTimeStep(void *GlobalData) const;
	
	PerfCounter cntr_;
	
protected:
	ThresholdLevelSetFunc();

};

}

//include implementation
#include "src/diffFunc.cxx"

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
