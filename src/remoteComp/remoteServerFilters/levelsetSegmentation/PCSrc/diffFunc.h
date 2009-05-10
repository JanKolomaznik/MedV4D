#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "itkFiniteDifferenceFunction.h"
#include "common/perfCounter.h"
#include "speedTermSolver.h"
//#include "advectionTermSolver.h"
#include "curvatureTermSolver.h"

namespace M4D {
namespace Cell {

template <class ImageType, class FeatureImageType = ImageType>
class ThresholdLevelSetFunc
	: public itk::FiniteDifferenceFunction<FeatureImageType>
	, public SpeedTermSolver<FeatureImageType, typename itk::FiniteDifferenceFunction<FeatureImageType>::NeighborhoodType, typename itk::FiniteDifferenceFunction<FeatureImageType>::FloatOffsetType>
	//, public AdvectionTermSolver,
	, public CurvatureTermSolver<FeatureImageType>
{
public:
	typedef ThresholdLevelSetFunc<ImageType, FeatureImageType> Self;
	typedef itk::SmartPointer<Self> Pointer;
	typedef itk::FiniteDifferenceFunction<ImageType> Superclass;
	typedef typename Superclass::PixelType 	PixelType;
	typedef typename Superclass::NeighborhoodType 	NeighborhoodType;
	typedef typename Superclass::FloatOffsetType 	FloatOffsetType;
	typedef typename Superclass::RadiusType 	RadiusType;
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::NeighborhoodScalesType NeighborhoodScalesType;
	typedef GlobalDataStruct<PixelType, FeatureImageType::ImageDimension> GlobalDataType;
	
	itkNewMacro(Self);
	
	virtual PixelType ComputeUpdate(
			const NeighborhoodType &neighborhood,
	        void *globalData,
	        const FloatOffsetType& offset = FloatOffsetType(0.0) );
	
	TimeStepType ComputeGlobalTimeStep(void *GlobalData) const;
	
	void *GetGlobalDataPointer() const { return new GlobalDataType(); }
	void ReleaseGlobalDataPointer(void *globalData) const { delete (GlobalDataType*) globalData; }
	
	PerfCounter cntr_;
	
protected:
	ThresholdLevelSetFunc();
	
private:
	/** Slices for the ND neighborhood. */
	  std::slice x_slice[FeatureImageType::ImageDimension];

	  /** The offset of the center pixel in the neighborhood. */
	  ::size_t m_Center;

	  /** Stride length along the y-dimension. */
	  ::size_t m_xStride[FeatureImageType::ImageDimension];
	  
	  /** Constants used in the time step calculation. */
	  double m_WaveDT;
	  double m_DT;


};

}
}
//include implementation
#include "src/diffFunc.tcc"

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
