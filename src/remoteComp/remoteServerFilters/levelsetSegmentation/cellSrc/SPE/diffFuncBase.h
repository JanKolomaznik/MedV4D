#ifndef DIFFFUNCBASE_H_
#define DIFFFUNCBASE_H_

#include "../commonConsts.h"

namespace itk {

template<class TNeighborhood>
class MyDiffFuncBase : public CommonTypes<TNeighborhood::ImageType::ImageDimension>
{
public:
  /** Standard class typedefs. */
  typedef MyDiffFuncBase      Self;
  typedef CommonTypes<TNeighborhood::ImageType::ImageDimension> SuperClass;

  /** Extract some parameters from the image type */
  typedef typename TNeighborhood::PixelType     PixelType;
  
  typedef typename TNeighborhood::ImageType ImageType;

  /** Neighborhood radius type */
  typedef typename TNeighborhood::RadiusType RadiusType;
  
  typedef typename SuperClass::NeighborhoodScalesType NeighborhoodScalesType;
  typedef typename SuperClass::TimeStepType TimeStepType;

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
    for( unsigned int i = 0; i < TNeighborhood::ImageType::ImageDimension; i++ )
      {
      m_ScaleCoefficients[i] = vals[i];
      }
    }
  
  const NeighborhoodScalesType ComputeNeighborhoodScales() const
  {
	  NeighborhoodScalesType neighborhoodScales;
	    neighborhoodScales.Fill(0.0);
	    typedef typename NeighborhoodScalesType::ComponentType NeighborhoodScaleType;
	    for(int i=0; i<TNeighborhood::ImageType::ImageDimension; i++)
	      {
	      if (this->m_Radius[i] > 0)
	        {
	        neighborhoodScales[i] = this->m_ScaleCoefficients[i] / this->m_Radius[i];
	        }
	      }
	    return neighborhoodScales;
  }

  /** Computes the time step for an update given a global data structure.
   * The data used in the computation may take different forms depending on
   * the nature of the equations.  This global data cannot be kept in the
   * instance of the equation object itself since the equation object must
   * remain stateless for thread safety.  The global data is therefore managed
   * for each thread by the finite difference solver filters. */
  virtual TimeStepType ComputeGlobalTimeStep(void *GlobalData) const =0;
  
protected:
  MyDiffFuncBase() 
  {
    // initialize variables
    m_Radius.Fill( 0 );
    for (unsigned int i = 0; i < TNeighborhood::ImageType::ImageDimension; i++)
      {
      m_ScaleCoefficients[i] = 1.0;
      }
  }
  ~MyDiffFuncBase() {}

  RadiusType m_Radius;
  NeighborhoodScalesType m_ScaleCoefficients;
};

}  // namespace itk

#endif /*DIFFFUNCBASE_H_*/
