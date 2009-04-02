#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#define CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_

#include "speedTermSolver.h"
//#include "advectionTermSolver.h"
#include "curvatureTermSolver.h"
#include "../commonConsts.h"

namespace itk
{

template <class TInputNeighbour, class TFeatureNeighbour = TInputNeighbour>
class ThresholdLevelSetFunc
	: public SpeedTermSolver<typename TFeatureNeighbour::ImageType, TInputNeighbour, typename CommonTypes<TInputNeighbour::ImageType::ImageDimension>::FloatOffsetType>
	//, public AdvectionTermSolver,
	, public CurvatureTermSolver<typename TInputNeighbour::ImageType>
	, public CommonTypes<TInputNeighbour::ImageType::ImageDimension>
{
public:
	typedef ThresholdLevelSetFunc<TInputNeighbour, TFeatureNeighbour> Self;
	typedef CommonTypes<TInputNeighbour::ImageType::ImageDimension> Superclass;
	typedef typename Superclass::FloatOffsetType 	FloatOffsetType;
	typedef typename Superclass::TimeStepType TimeStepType;
	typedef typename Superclass::NeighborhoodScalesType NeighborhoodScalesType;	
	  typedef typename TInputNeighbour::PixelType     PixelType;	  
	  typedef typename TInputNeighbour::RadiusType RadiusType;
	
	typedef TInputNeighbour NeighborhoodType;
	typedef typename TInputNeighbour::ImageType ImageType;
	typedef GlobalDataStruct<PixelType, ImageType::ImageDimension> GlobalDataType;

	
	virtual PixelType ComputeUpdate(
			const NeighborhoodType &neighborhood,
	        void *globalData,
	        const FloatOffsetType& offset = FloatOffsetType(0.0) );
	
	TimeStepType ComputeGlobalTimeStep(void *GlobalData) const;

	/** Sets the radius of the neighborhood this MyDiffFuncBase
	   * needs to perform its calculations. */
	  void SetRadius(const RadiusType &r)
	    { m_Radius = r; }

	  /** Returns the radius of the neighborhood this MyDiffFuncBase
	   * needs to perform its calculations. */
	  const RadiusType &GetRadius() const
	    { return m_Radius; }

	  /** Set the ScaleCoefficients for the difference
	   * operators. The defaults a 1.0. These can be set to take the image
	   * spacing into account. */
	  void SetScaleCoefficients (NeighborhoodScalesType vals)
	    {
	    for( unsigned int i = 0; i < ImageType::ImageDimension; i++ )
	      {
	      m_ScaleCoefficients[i] = vals[i];
	      }
	    }
	  
	  const NeighborhoodScalesType ComputeNeighborhoodScales() const
	  {
		  NeighborhoodScalesType neighborhoodScales;
		    neighborhoodScales.Fill(0.0);
		    typedef typename NeighborhoodScalesType::ComponentType NeighborhoodScaleType;
		    for(int i=0; i<ImageType::ImageDimension; i++)
		      {
		      if (this->m_Radius[i] > 0)
		        {
		        neighborhoodScales[i] = this->m_ScaleCoefficients[i] / this->m_Radius[i];
		        }
		      }
		    return neighborhoodScales;
	  }
	
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
	  
	  RadiusType m_Radius;
	  NeighborhoodScalesType m_ScaleCoefficients;
};

}

//include implementation
#include "src/diffFunc.tcc"

#endif /*CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_*/
