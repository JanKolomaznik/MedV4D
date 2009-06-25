
#include "common/Types.h"
#include "../speedTermSolver.h"
#include <math.h>
#include "../../vnl_math.h"

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////

SpeedTermSolver
::SpeedTermSolver()
{
	this->CountMiddleVal();
	m_PropagationWeight = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////

SpeedTermSolver::FeatureScalarType
SpeedTermSolver
	::GetSpeedInPoint(const FeatureScalarType &pixelValue) const
{
    if (pixelValue < m_threshIntervalMid)
      {
      return pixelValue - m_LowerThreshold;
      }
    else
      {
    	return m_UpperThreshold - pixelValue;
      }
}
	
///////////////////////////////////////////////////////////////////////////////

SpeedTermSolver::FeatureScalarType
SpeedTermSolver
::PropagationSpeed(const TFeatureNeighbourhoodIter &neighborhood,
                   const ContinuousIndexType &offset) const
{
  
  FeatureScalarType val = Interpolate(offset, neighborhood);

  return val;
}

///////////////////////////////////////////////////////////////////////////////

SpeedTermSolver::FeatureScalarType
SpeedTermSolver
	::Interpolate(const ContinuousIndexType &index, const TFeatureNeighbourhoodIter &neighb) const
{
	unsigned int dim;  // index over dimension
	
	unsigned int neighborCount = 1 << DIM;
	
	typedef float32 RealType;
	
	TContinuousIndexDouble cdx;
	for (unsigned i = 0; i < DIM; ++i)
	    {
	    cdx[i] = static_cast<double>(neighb.GetNeighborhood().m_currIndex[i]) - index[i];
	    }
	
	if(! neighb.GetNeighborhood().IsWithinImageContinuos(cdx))
	{
		return GetSpeedInPoint(neighb.GetCenterPixel());
	}

	  /**
	   * Compute base index = closet index below point
	   * Compute distance from point to base index
	   */
	  signed long baseIndex[DIM];
	  double distance[DIM];
	  long tIndex;

	  for( dim = 0; dim < DIM; dim++ )
	    {
	    // The following "if" block is equivalent to the following line without
	    // having to call floor.
	    //    baseIndex[dim] = (long) vcl_floor(index[dim] );
	    if (cdx[dim] >= 0.0)
	      {
	      baseIndex[dim] = (long) cdx[dim];
	      }
	    else
	      {
	      tIndex = (long) cdx[dim];
	      if (double(tIndex) != cdx[dim])
	        {
	        tIndex--;
	        }
	      baseIndex[dim] = tIndex;
	      }
	    distance[dim] = cdx[dim] - double( baseIndex[dim] );
	    }
	  
	  /**
	   * Interpolated value is the weighted sum of each of the surrounding
	   * neighbors. The weight for each neighbor is the fraction overlap
	   * of the neighbor pixel with respect to a pixel centered on point.
	   */
	  RealType value = 0;//itk::NumericTraits<RealType>::Zero;
	  RealType totalOverlap = 0;//itk::NumericTraits<RealType>::Zero;

	  for( unsigned int counter = 0; counter < neighborCount; counter++ )
	    {

	    double overlap = 1.0;          // fraction overlap
	    unsigned int upper = counter;  // each bit indicates upper/lower neighbour
	    TOffset neighIndex;

	    // get neighbor index and overlap fraction
	    for( dim = 0; dim < DIM; dim++ )
	      {

	      if ( upper & 1 )
	        {
	        neighIndex[dim] = baseIndex[dim] + 1;
	        overlap *= distance[dim];
	        }
	      else
	        {
	        neighIndex[dim] = baseIndex[dim];
	        overlap *= 1.0 - distance[dim];
	        }

	      upper >>= 1;

	      }
	    
	    // workaround to guarantee index coord vals be from <-1,1> interval
//	    for(uint32 i=0; i<DIM; i++)
//	    {
//	    	if(neighIndex[i] < -1) neighIndex[i] = -1;
//	    	if(neighIndex[i] > 1) neighIndex[i] = 1;
//	    }
	    
	    // get neighbor value only if overlap is not zero
	    if( overlap )
	      {
	    	TOffset o = neighIndex - neighb.GetNeighborhood().m_currIndex;
	    	float pixel = GetSpeedInPoint(neighb.GetPixel(o));
	    		//D_PRINT("Speed at" << neighIndex << "=" << pixel);
	      value += overlap * static_cast<RealType>( pixel );
	      totalOverlap += overlap;
	      }

	    if( totalOverlap == 1.0 )
	      {
	      // finished
	      break;
	      }

	    }

	  return ( static_cast<FeatureScalarType>( value ) );
}

///////////////////////////////////////////////////////////////////////////////

SpeedTermSolver::FeatureScalarType
SpeedTermSolver
::ComputePropagationTerm(
	  const TFeatureNeighbourhoodIter &neighborhood,
	  const ContinuousIndexType& offset,
	  GlobalDataType *gd)
{
	const FeatureScalarType ZERO = 0;//itk::NumericTraits<FeatureScalarType>::Zero;
	uint32 i;
	
	if(m_PropagationWeight == 0)
		return ZERO;

	// Get the propagation speed
	FeatureScalarType propagation_term = 
		m_PropagationWeight * Interpolate(offset, neighborhood);//PropagationSpeed(neighborhood, offset);

	//
	// Construct upwind gradient values for use in the propagation speed term:
	//  $\beta G(\mathbf{x})\mid\nabla\phi\mid$
	//
	// The following scheme for ``upwinding'' in the normal direction is taken
	// from Sethian, Ch. 6 as referenced above.
	//
	FeatureScalarType propagation_gradient = ZERO;

	if ( propagation_term> ZERO )
	{
		for(i = 0; i< DIM; i++)
		{
			propagation_gradient += vnl_math_sqr( vnl_math_max(gd->m_dx_backward[i], ZERO) )
			+ vnl_math_sqr( vnl_math_min(gd->m_dx_forward[i], ZERO) );
		}
	}
	else
	{
		for(i = 0; i< DIM; i++)
		{
			propagation_gradient += vnl_math_sqr( vnl_math_min(gd->m_dx_backward[i], ZERO) )
			+ vnl_math_sqr( vnl_math_max(gd->m_dx_forward[i], ZERO) );
		}
	}

	// Collect energy change from propagation term.  This will be used in
	// calculating the maximum time step that can be taken for this iteration.
	gd->m_MaxPropagationChange =
	vnl_math_max(gd->m_MaxPropagationChange,
			vnl_math_abs(propagation_term));

	propagation_term *= sqrt( propagation_gradient );
	return propagation_term;
}

///////////////////////////////////////////////////////////////////////////////

