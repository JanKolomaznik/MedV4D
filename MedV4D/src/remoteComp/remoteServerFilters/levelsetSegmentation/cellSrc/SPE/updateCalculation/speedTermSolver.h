#ifndef SPEEDTERMSOLVER_H_
#define SPEEDTERMSOLVER_H_

#include "globalData.h"
#include "../tools/neighbourhoodIterator.h"

namespace M4D
{
namespace Cell
{

class SpeedTermSolver
{
public:
	typedef TIndex IndexType;
	typedef TPixelValue FeaturePixelType;
	typedef FeaturePixelType FeatureScalarType;
	typedef GlobalDataStruct GlobalDataType;
	typedef TContinuousIndex ContinuousIndexType;
	typedef NeighbourIteratorCell<FeaturePixelType> TFeatureNeighbourhoodIter;

	/** Set/Get threshold values */
	void SetUpperThreshold(FeatureScalarType f)
	{
		m_UpperThreshold = f;
		CountMiddleVal();
	}
	FeatureScalarType GetUpperThreshold() const
	{
		return m_UpperThreshold;
	}
	void SetLowerThreshold(FeatureScalarType f)
	{
		m_LowerThreshold = f;
		CountMiddleVal();
	}
	FeatureScalarType GetLowerThreshold() const
	{
		return m_LowerThreshold;
	}

	/** Beta.  Scales all propagation term values. */
	void SetPropagationWeight(const float32 p)
	{
		m_PropagationWeight = p;
	}
	float32 GetPropagationWeight() const
	{
		return m_PropagationWeight;
	}

protected:

	FeatureScalarType m_UpperThreshold;
	FeatureScalarType m_LowerThreshold;
	FeatureScalarType m_threshIntervalMid;

	float32 m_PropagationWeight;

	FeatureScalarType ComputePropagationTerm(
			const TFeatureNeighbourhoodIter &neighborhood,
			const ContinuousIndexType& offset, GlobalDataType *gd);

	FeatureScalarType GetSpeedInPoint(const FeatureScalarType &pixelValue) const;

	FeatureScalarType PropagationSpeed(
			const TFeatureNeighbourhoodIter &neighborhood,
			const ContinuousIndexType &offset) const;

	SpeedTermSolver();

private:

	inline void CountMiddleVal(void)
	{
		m_threshIntervalMid = ( (m_UpperThreshold - m_LowerThreshold) / 2.0 )
				+ m_LowerThreshold;
	}

	FeatureScalarType Interpolate(const ContinuousIndexType &index,
			const TFeatureNeighbourhoodIter &neighb) const;
};

}
}

#endif /*THRESHOLDINGSPEEDFUNCTION_H_*/
