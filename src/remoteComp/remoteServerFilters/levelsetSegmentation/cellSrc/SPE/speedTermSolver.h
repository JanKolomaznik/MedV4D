#ifndef SPEEDTERMSOLVER_H_
#define SPEEDTERMSOLVER_H_

#include "globalData.h"

namespace M4D {
namespace Cell {

template<typename FeatureScalarType, typename TFeatureNeighbourhood>
class SpeedTermSolver
{
public:
	typedef typename TFeatureNeighbourhood::IndexType IndexType;
	typedef typename TFeatureNeighbourhood::PixelType FeaturePixelType;
	
	typedef GlobalDataStruct<FeaturePixelType, TFeatureNeighbourhood::Dim> GlobalDataType;
	typedef typename TFeatureNeighbourhood::ContinuousIndexType ContinuousIndexType;
	typedef ContinuousIndexType FloatOffsetType;
	
	/** Set/Get threshold values */
	  void SetUpperThreshold(FeatureScalarType f)
	  { m_UpperThreshold = f; CountMiddleVal(); }
	  FeatureScalarType GetUpperThreshold() const
	  { return m_UpperThreshold; }
	  void SetLowerThreshold(FeatureScalarType f)
	  { m_LowerThreshold = f; CountMiddleVal(); }
	  FeatureScalarType GetLowerThreshold() const
	  { return m_LowerThreshold; }
	  
	  /** Beta.  Scales all propagation term values. */
      void SetPropagationWeight(const float32 p)
        { m_PropagationWeight = p; }
      float32 GetPropagationWeight() const
        { return m_PropagationWeight; }
	  
protected:
	
  FeatureScalarType m_UpperThreshold;
  FeatureScalarType m_LowerThreshold;	  
  FeatureScalarType m_threshIntervalMid;
  
  float32 m_PropagationWeight;
  
  FeatureScalarType ComputePropagationTerm(
		  const TFeatureNeighbourhood &neighborhood,
		  const FloatOffsetType& offset,
		  GlobalDataType *gd) const;
  
  FeatureScalarType GetSpeedInPoint(const FeatureScalarType &pixelValue) const;
  
  FeatureScalarType PropagationSpeed(const TFeatureNeighbourhood &neighborhood,
                     const FloatOffsetType &offset) const;
  
  SpeedTermSolver();
  
private:
	
	inline void CountMiddleVal(void)
	{
		m_threshIntervalMid = 
			( (m_UpperThreshold - m_LowerThreshold) / 2.0 ) 
			+ m_LowerThreshold;
	}
	
	FeatureScalarType Interpolate(ContinuousIndexType &index, const TFeatureNeighbourhood &neighb) const;
};

//include implementation
#include "src/speedTermSolver.tcc"

}}

#endif /*THRESHOLDINGSPEEDFUNCTION_H_*/
