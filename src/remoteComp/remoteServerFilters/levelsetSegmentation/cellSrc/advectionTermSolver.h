#ifndef ADVECTIONTERMSOLVER_H_
#define ADVECTIONTERMSOLVER_H_

#include "globalData.h"

namespace itk
{

template <class TImageType, typename NeighborhoodType, typename PixelType, typename FloatOffsetType>
class AdvectionTermSolver
{
public:
	  /** Alpha.  Scales all advection term values.*/ 
    void SetAdvectionWeight(const ScalarValueType a)
      { m_AdvectionWeight = a; }
    ScalarValueType GetAdvectionWeight() const
      { return m_AdvectionWeight; }
	    
protected:
	typename PixelType ComputeAdvectionTerm(void);
	
	VectorType AdvectionField(const typename NeighborhoodType &neighborhood,
	                 const typename FloatOffsetType &offset, GlobalDataStruct *)  const;
	
private:
	float32 m_AdvectionWeight;
};

}

//include implementation
#include "src/advectionTermSolver.tcc"

#endif /*ADVECTIONTERMSOLVER_H_*/
