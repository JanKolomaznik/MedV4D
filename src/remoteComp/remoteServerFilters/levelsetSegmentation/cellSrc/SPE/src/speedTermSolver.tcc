#ifndef SPEEDTERMSOLVER_H_
#error File SpeedTermSolver.tcc cannot be included directly!
#else

#include <math.h>

///////////////////////////////////////////////////////////////////////////////

template<typename FeatureScalarType, typename TFeatureNeighbourhood>
SpeedTermSolver<FeatureScalarType, TFeatureNeighbourhood>
::SpeedTermSolver()
{
	this->CountMiddleVal();
	m_PropagationWeight = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////

template<typename FeatureScalarType, typename TFeatureNeighbourhood>
FeatureScalarType
SpeedTermSolver<FeatureScalarType, TFeatureNeighbourhood>
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

template<typename FeatureScalarType, typename TFeatureNeighbourhood>
FeatureScalarType
SpeedTermSolver<FeatureScalarType, TFeatureNeighbourhood>
::PropagationSpeed(const TFeatureNeighbourhood &neighborhood,
                   const FloatOffsetType &offset) const
{
  const IndexType idx = neighborhood.GetIndex();

  ContinuousIndexType cdx;
  for (unsigned i = 0; i < TFeatureNeighbourhood::Dim; ++i)
    {
    cdx[i] = static_cast<double>(idx[i]) - offset[i];
    }
  
  FeatureScalarType val = Interpolate(cdx, neighborhood);
  //LOG("ComputePropagationTerm at index: " << cdx << ", " << val);

	return val;
}

///////////////////////////////////////////////////////////////////////////////

template<typename FeatureScalarType, typename TFeatureNeighbourhood>
FeatureScalarType
SpeedTermSolver<FeatureScalarType, TFeatureNeighbourhood>
	::Interpolate(ContinuousIndexType &index, const TFeatureNeighbourhood &neighb) const
{
	unsigned int dim;  // index over dimension
	
	unsigned int neighborCount = 1 << TFeatureNeighbourhood::Dim;
	
	typedef float32 RealType;

	  /**
	   * Compute base index = closet index below point
	   * Compute distance from point to base index
	   */
	  signed long baseIndex[TFeatureNeighbourhood::Dim];
	  double distance[TFeatureNeighbourhood::Dim];
	  long tIndex;

	  for( dim = 0; dim < TFeatureNeighbourhood::Dim; dim++ )
	    {
	    // The following "if" block is equivalent to the following line without
	    // having to call floor.
	    //    baseIndex[dim] = (long) vcl_floor(index[dim] );
	    if (index[dim] >= 0.0)
	      {
	      baseIndex[dim] = (long) index[dim];
	      }
	    else
	      {
	      tIndex = (long) index[dim];
	      if (double(tIndex) != index[dim])
	        {
	        tIndex--;
	        }
	      baseIndex[dim] = tIndex;
	      }
	    distance[dim] = index[dim] - double( baseIndex[dim] );
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
	    IndexType neighIndex;

	    // get neighbor index and overlap fraction
	    for( dim = 0; dim < TFeatureNeighbourhood::Dim; dim++ )
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
	    
	    // get neighbor value only if overlap is not zero
	    if( overlap )
	      {
	      value += overlap * static_cast<RealType>( GetSpeedInPoint(neighb.GetPixel(neighIndex)) );
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

template<typename FeatureScalarType, typename TFeatureNeighbourhood>
FeatureScalarType
SpeedTermSolver<FeatureScalarType, TFeatureNeighbourhood>
::ComputePropagationTerm(
	  const TFeatureNeighbourhood &neighborhood,
	  const FloatOffsetType& offset,
	  GlobalDataType *gd) const
{
	const FeatureScalarType ZERO = 0;//itk::NumericTraits<FeatureScalarType>::Zero;
	uint32 i;
	
	if(m_PropagationWeight == 0)
		return ZERO;

	// Get the propagation speed
	FeatureScalarType propagation_term = 
		m_PropagationWeight * PropagationSpeed(neighborhood, offset);

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
		for(i = 0; i< TFeatureNeighbourhood::Dim; i++)
		{
			propagation_gradient += vnl_math_sqr( vnl_math_max(gd->m_dx_backward[i], ZERO) )
			+ vnl_math_sqr( vnl_math_min(gd->m_dx_forward[i], ZERO) );
		}
	}
	else
	{
		for(i = 0; i< TFeatureNeighbourhood::Dim; i++)
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

#endif
