#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "speedTermSolver.h"
//#include "advectionTermSolver.h"
#include "curvatureTermSolver.h"
#include "neighbourhoodIterator.h"
//#include "../commonConsts.h"

namespace M4D {
namespace Cell {

class ThresholdLevelSetFunc
	: public SpeedTermSolver	//<typename TInputNeighbour::PixelType, TInputNeighbour>
	//, public AdvectionTermSolver,
	, public CurvatureTermSolver	//<typename TInputNeighbour::PixelType, TInputNeighbour::Dim>
	//, public CommonTypes<TInputNeighbour::Dim>
{
public:
//	typedef ThresholdLevelSetFunc<TInputNeighbour, TFeatureNeighbour> Self;
//	typedef CommonTypes<TInputNeighbour::Dim> Superclass;
//	typedef typename Superclass::FloatOffsetType 	FloatOffsetType;
//	typedef typename Superclass::TimeStepType TimeStepType;
//	typedef typename Superclass::NeighborhoodScalesType NeighborhoodScalesType;
//	  typedef typename TInputNeighbour::PixelType     PixelType;
//	  typedef typename TInputNeighbour::RadiusType RadiusType;
//	
//	typedef TInputNeighbour NeighborhoodType;
//	typedef GlobalDataStruct<PixelType, TInputNeighbour::Dim> GlobalDataType;
	typedef NeighbourIteratorCell NeighborhoodIteratorType;

	
	TPixelValue ComputeUpdate(
			const NeighborhoodIteratorType &neighborhood,
			const NeighborhoodIteratorType &featureNeib,
			GlobalDataStruct *globalData,
	        const TContinuousIndex& offset );
	
	TimeStepType ComputeGlobalTimeStep(void *GlobalData);

	/** Sets the radius of the neighborhood this MyDiffFuncBase
	   * needs to perform its calculations. */
	  void SetRadius(const TRadius &r)
	    { m_Radius = r; }

	  /** Returns the radius of the neighborhood this MyDiffFuncBase
	   * needs to perform its calculations. */
	  const TRadius &GetRadius() const
	    { return m_Radius; }

	  /** Set the ScaleCoefficients for the difference
	   * operators. The defaults a 1.0. These can be set to take the image
	   * spacing into account. */
	  void SetScaleCoefficients (TNeighborhoodScales vals)
	    {
	    for( unsigned int i = 0; i < DIM; i++ )
	      {
	      m_ScaleCoefficients[i] = vals[i];
	      }
	    }
	  
	  const TNeighborhoodScales ComputeNeighborhoodScales()
	  {
		  TNeighborhoodScales neighborhoodScales;
		    
		    for(int i=0; i<DIM; i++)
		      {
		      if (this->m_Radius[i] > 0)
		        {
		        neighborhoodScales[i] = this->m_ScaleCoefficients[i] / this->m_Radius[i];
		        }
		      }
		    return neighborhoodScales;
	  }
	
	ThresholdLevelSetFunc();
	virtual ~ThresholdLevelSetFunc() {}
	
private:
//	/** Slices for the ND neighborhood. */
//	  std::slice x_slice[TInputNeighbour::Dim];
//
//	  /** The offset of the center pixel in the neighborhood. */
//	  ::size_t m_Center;
//
//	  /** Stride length along the y-dimension. */
//	  ::size_t m_xStride[TInputNeighbour::Dim];
	  
	  /** Constants used in the time step calculation. */
	  double m_WaveDT;
	  double m_DT;
	  
	  TRadius m_Radius;
	  TNeighborhoodScales m_ScaleCoefficients;
};

//include implementation
//#include "src/diffFunc.tcc"
	
}}

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
