#ifndef SPEEDTERMSOLVER_H_
#define SPEEDTERMSOLVER_H_

#include "globalData.h"
#include "itkLinearInterpolateImageFunction.h"

namespace itk
{

template<class FeatureImageType, typename NeighborhoodType, typename FloatOffsetType>
class SpeedTermSolver
{
public:
	typedef typename FeatureImageType::PixelType FeatureScalarType;
	typedef typename NeighborhoodType::IndexType IndexType;
	typedef LinearInterpolateImageFunction<FeatureImageType>  InterpolatorType;
	
	typedef GlobalDataStruct<FeatureScalarType, FeatureImageType::ImageDimension> GlobalDataType;
	typedef typename InterpolatorType::ContinuousIndexType ContinuousIndexType;
	
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
      
  	void SetFeatureImage(const FeatureImageType *featureIm) { 
  		m_featureImage = featureIm; 
  		m_Interpolator->SetInputImage(featureIm); 
  		}
	  
protected:
	
  FeatureScalarType m_UpperThreshold;
  FeatureScalarType m_LowerThreshold;	  
  FeatureScalarType m_threshIntervalMid;
  
  float32 m_PropagationWeight;
  
  FeatureScalarType ComputePropagationTerm(
		  const NeighborhoodType &neighborhood,
		  const FloatOffsetType& offset,
		  GlobalDataType *gd) const;
  
  FeatureScalarType GetSpeedInPoint(const IndexType &index) const;
  
  FeatureScalarType PropagationSpeed(const NeighborhoodType &neighborhood,
                     const FloatOffsetType &offset, GlobalDataType *gd) const;
  
  SpeedTermSolver();
  
private:
	const FeatureImageType *m_featureImage;
  
	inline void CountMiddleVal(void)
	{
		m_threshIntervalMid = 
			( (m_UpperThreshold - m_LowerThreshold) / 2.0 ) 
			+ m_LowerThreshold;
	}
	
	FeatureScalarType Interpolate(ContinuousIndexType &index) const;
	
	typename InterpolatorType::Pointer m_Interpolator;
};

}

//include implementation
#include "src/speedTermSolver.tcc"

#endif /*THRESHOLDINGSPEEDFUNCTION_H_*/
