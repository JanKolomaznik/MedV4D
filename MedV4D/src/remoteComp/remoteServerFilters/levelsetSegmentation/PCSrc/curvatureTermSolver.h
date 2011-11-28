#ifndef CURVATURETERMSOLVER_H_
#define CURVATURETERMSOLVER_H_

#include "globalData.h"

namespace M4D {
namespace Cell {

template< class ImageType > 
class CurvatureTermSolver
{
public:
	typedef typename ImageType::PixelType ScalarValueType;
	typedef GlobalDataStruct<ScalarValueType, ImageType::ImageDimension> GlobalDataType;
	
    /** Gamma. Scales all curvature weight values */
    void SetCurvatureWeight(const ScalarValueType c)
      { m_CurvatureWeight = c; }
    ScalarValueType GetCurvatureWeight() const
      { return m_CurvatureWeight; }
	    
protected:
	ScalarValueType ComputeCurvatureTerm(GlobalDataType *gd);
	
private:
	ScalarValueType m_CurvatureWeight;
	
	ScalarValueType ComputeMeanCurvature(GlobalDataType *gd);
};

}
}
#include "src/curvatureTermSolver.tcc"

#endif /*CURVATURETERMSOLVER_H_*/
