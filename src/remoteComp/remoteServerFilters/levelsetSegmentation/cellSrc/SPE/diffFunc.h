#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "diffFuncBase.h"
#include "common/perfCounter.h"
#include "speedTermSolver.h"
//#include "advectionTermSolver.h"
#include "curvatureTermSolver.h"

namespace itk
{

template <class TInputNeighbour, class TFeatureNeighbour = TInputNeighbour>
class ThresholdLevelSetFunc
	: public MyDiffFuncBase<TInputNeighbour>
	, public SpeedTermSolver<typename TFeatureNeighbour::ImageType, TInputNeighbour, typename MyDiffFuncBase<TInputNeighbour>::FloatOffsetType>
	//, public AdvectionTermSolver,
	, public CurvatureTermSolver<typename TInputNeighbour::ImageType>
{
public:
	typedef ThresholdLevelSetFunc<TInputNeighbour, TFeatureNeighbour> Self;
	typedef MyDiffFuncBase<TInputNeighbour> Superclass;
	typedef typename Superclass::PixelType 	PixelType;
	typedef typename Superclass::FloatOffsetType 	FloatOffsetType;
	typedef typename Superclass::RadiusType 	RadiusType;
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::NeighborhoodScalesType NeighborhoodScalesType;
	
	typedef TInputNeighbour NeighborhoodType;
	typedef typename TInputNeighbour::ImageType ImageType;
	typedef GlobalDataStruct<PixelType, ImageType::ImageDimension> GlobalDataType;

	
	virtual PixelType ComputeUpdate(
			const NeighborhoodType &neighborhood,
	        void *globalData,
	        const FloatOffsetType& offset = FloatOffsetType(0.0) );
	
	TimeStepType ComputeGlobalTimeStep(void *GlobalData) const;
	
	void *GetGlobalDataPointer() const { return new GlobalDataType(); }
	void ReleaseGlobalDataPointer(void *globalData) const { delete (GlobalDataType*) globalData; }
	
	ThresholdLevelSetFunc();
	
private:
	/** Slices for the ND neighborhood. */
	  std::slice x_slice[ImageType::ImageDimension];

	  /** The offset of the center pixel in the neighborhood. */
	  ::size_t m_Center;

	  /** Stride length along the y-dimension. */
	  ::size_t m_xStride[ImageType::ImageDimension];
	  
	  /** Constants used in the time step calculation. */
	  double m_WaveDT;
	  double m_DT;


};

}

//include implementation
#include "src/diffFunc.tcc"

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
