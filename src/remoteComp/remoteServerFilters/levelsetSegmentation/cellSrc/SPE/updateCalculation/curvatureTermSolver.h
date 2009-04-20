#ifndef CURVATURETERMSOLVER_H_
#define CURVATURETERMSOLVER_H_

#include "globalData.h"

namespace M4D {
namespace Cell {

class CurvatureTermSolver
{
public:
	typedef TPixelValue ScalarValueType;
	
    /** Gamma. Scales all curvature weight values */
    void SetCurvatureWeight(const ScalarValueType c)
      { m_CurvatureWeight = c; }
    ScalarValueType GetCurvatureWeight() const
      { return m_CurvatureWeight; }
	    
protected:
	ScalarValueType ComputeCurvatureTerm(GlobalDataStruct *gd);
	
private:
	ScalarValueType m_CurvatureWeight;
	
	ScalarValueType ComputeMeanCurvature(GlobalDataStruct *gd);
};

}}

#endif /*CURVATURETERMSOLVER_H_*/
