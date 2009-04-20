#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "speedTermSolver.h"
#include "curvatureTermSolver.h"
#include "../tools/neighbourhoodIterator.h"

namespace M4D {
namespace Cell {

class ThresholdLevelSetFunc
	: public SpeedTermSolver
	, public CurvatureTermSolver
{
public:
	typedef NeighbourIteratorCell<TPixelValue> NeighborhoodIteratorType;
	
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
	~ThresholdLevelSetFunc() {}
	
private:
	  
	  /** Constants used in the time step calculation. */
	  double m_WaveDT;
	  double m_DT;
	  
	  TRadius m_Radius;
	  TNeighborhoodScales m_ScaleCoefficients;
};
	
}
}

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
