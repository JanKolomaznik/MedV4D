#ifndef SPEEDTERMSOLVER_H_
#error File SpeedTermSolver.tcc cannot be included directly!
#else

namespace itk
{

///////////////////////////////////////////////////////////////////////////////

template<class FeatureImageType, typename NeighborhoodType, typename FloatOffsetType>
SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>
::SpeedTermSolver()
{
	this->CountMiddleVal();
	m_PropagationWeight = 1.0f;
	m_Interpolator = InterpolatorType::New();
}

///////////////////////////////////////////////////////////////////////////////

template<class FeatureImageType, typename NeighborhoodType, typename FloatOffsetType>
typename SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>::FeatureScalarType
SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>
	::GetSpeedInPoint(const IndexType &index) const
{
	FeatureScalarType pixelValue = m_featureImage->GetPixel(index);
	
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

template<class FeatureImageType, typename NeighborhoodType, typename FloatOffsetType>
typename SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>::FeatureScalarType
SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>
::PropagationSpeed(const NeighborhoodType &neighborhood,
                   const FloatOffsetType &offset, GlobalDataType *gd) const
{
  const IndexType idx = neighborhood.GetIndex();  

  ContinuousIndexType cdx;
  for (unsigned i = 0; i < FeatureImageType::ImageDimension; ++i)
    {
    cdx[i] = static_cast<double>(idx[i]) - offset[i];
    }
  
  FeatureScalarType val = Interpolate(cdx);
  //LOUT << "ComputePropagationTerm at index: " << cdx << ", " << val << std::endl;

	return val;
}

///////////////////////////////////////////////////////////////////////////////

template<class FeatureImageType, typename NeighborhoodType, typename FloatOffsetType>
typename SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>::FeatureScalarType
SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>
	::Interpolate(ContinuousIndexType &index) const
{
	unsigned int dim;  // index over dimension
	
	unsigned int neighborCount = 1 << FeatureImageType::ImageDimension;
	
	typedef float32 RealType;

	  /**
	   * Compute base index = closet index below point
	   * Compute distance from point to base index
	   */
	  signed long baseIndex[FeatureImageType::ImageDimension];
	  double distance[FeatureImageType::ImageDimension];
	  long tIndex;

	  for( dim = 0; dim < FeatureImageType::ImageDimension; dim++ )
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
	  RealType value = NumericTraits<RealType>::Zero;
	  RealType totalOverlap = NumericTraits<RealType>::Zero;

	  for( unsigned int counter = 0; counter < neighborCount; counter++ )
	    {

	    double overlap = 1.0;          // fraction overlap
	    unsigned int upper = counter;  // each bit indicates upper/lower neighbour
	    IndexType neighIndex;

	    // get neighbor index and overlap fraction
	    for( dim = 0; dim < FeatureImageType::ImageDimension; dim++ )
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
	      value += overlap * static_cast<RealType>( GetSpeedInPoint(neighIndex) );
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

template<class FeatureImageType, typename NeighborhoodType, typename FloatOffsetType>
typename SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>::FeatureScalarType
SpeedTermSolver<FeatureImageType, NeighborhoodType, FloatOffsetType>
::ComputePropagationTerm(
	  const NeighborhoodType &neighborhood,
	  const FloatOffsetType& offset,
	  GlobalDataType *gd) const
{
	const FeatureScalarType ZERO = NumericTraits<FeatureScalarType>::Zero;
	uint32 i;
	
	if(m_PropagationWeight == 0)
		return ZERO;

	// Get the propagation speed
	FeatureScalarType propagation_term = 
		m_PropagationWeight * PropagationSpeed(neighborhood, offset, gd);

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
		for(i = 0; i< FeatureImageType::ImageDimension; i++)
		{
			propagation_gradient += vnl_math_sqr( vnl_math_max(gd->m_dx_backward[i], ZERO) )
			+ vnl_math_sqr( vnl_math_min(gd->m_dx_forward[i], ZERO) );
		}
	}
	else
	{
		for(i = 0; i< FeatureImageType::ImageDimension; i++)
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

	propagation_term *= vcl_sqrt( propagation_gradient );
	return propagation_term;
}

///////////////////////////////////////////////////////////////////////////////

}

#endif