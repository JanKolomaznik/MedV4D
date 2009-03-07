#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#error File diffFunc.cxx cannot be included directly!
#else

namespace itk
{

///////////////////////////////////////////////////////////////////////////////

template <class TImageType, class TFeatureImageType>
ThresholdLevelSetFunc<TImageType, TFeatureImageType>
::ThresholdLevelSetFunc()
{
	RadiusType radius;
	radius[0] = radius[1] = radius[2] = 1; 
	this->SetRadius(radius);
	
	cntr_.Reset();
}

///////////////////////////////////////////////////////////////////////////////

template <class TImageType, class TFeatureImageType>
typename ThresholdLevelSetFunc<TImageType, TFeatureImageType>::PixelType
ThresholdLevelSetFunc<TImageType, TFeatureImageType>
	::ComputeUpdate(const NeighborhoodType &neighborhood, void *globalData,
            const FloatOffsetType& offset)
{
	typedef LevelSetFunction<TImageType> LSFunc;
	typedef typename LSFunc::NeighborhoodType LSNeighborhoodType;
	typedef typename LSFunc::FloatOffsetType LSFloatOffsetType;
	
	cntr_.Start();
	PixelType retval = LevelSetFunction<TImageType>::ComputeUpdate(
			(const LSNeighborhoodType &) neighborhood,
			globalData,
			(const LSFloatOffsetType &) offset);
	cntr_.Stop();
	
	return retval;
}

///////////////////////////////////////////////////////////////////////////////

//template <class TImageType, class TFeatureImageType>
//typename ThresholdLevelSetFunc<TImageType, TFeatureImageType>::ScalarValueType
//ThresholdLevelSetFunc<TImageType, TFeatureImageType>
//	::GetSpeedInPoint()
//{
//	ScalarValueType upper_threshold = static_cast<ScalarValueType>(m_UpperThreshold);
//	ScalarValueType lower_threshold = static_cast<ScalarValueType>(m_LowerThreshold);
//	// count middle value
//	ScalarValueType mid = ( (upper_threshold - lower_threshold) / 2.0 ) + lower_threshold;
//	
//	  ScalarValueType threshold;
//	  for ( fit.GoToBegin(), sit.GoToBegin(); ! fit.IsAtEnd(); ++sit, ++fit) // for whole image
//	    {
//	    if (static_cast<ScalarValueType>(fit.Get()) < mid)
//	      {
//	      threshold = fit.Get() - lower_threshold;
//	      }
//	    else
//	      {
//	      threshold = upper_threshold - fit.Get();
//	      }
//	    
//	    if ( m_EdgeWeight != 0.0)
//	      {
//	      sit.Set( static_cast<ScalarValueType>(threshold + m_EdgeWeight * lit.Get()) );
//	      ++lit;
//	      }
//	    else
//	      {
//	      sit.Set( static_cast<ScalarValueType>(threshold) );
//	      }
//	    }
//	  
//	  return threshold;
//}
//
/////////////// From Segmentation function //////////////////
//
//template <class TImageType, class TFeatureImageType>
//typename SegmentationLevelSetFunction<TImageType, TFeatureImageType>::ScalarValueType
//SegmentationLevelSetFunction<TImageType, TFeatureImageType>
//::PropagationSpeed(const NeighborhoodType &neighborhood,
//                   const FloatOffsetType &offset, GlobalDataStruct *) const
//{
//  IndexType idx = neighborhood.GetIndex();
//
//  ContinuousIndexType cdx;
//  for (unsigned i = 0; i < ImageDimension; ++i)
//    {
//    cdx[i] = static_cast<double>(idx[i]) - offset[i];
//    }
//
//  if ( m_Interpolator->IsInsideBuffer(cdx) )
//    {
//    return (static_cast<ScalarValueType>(
//              m_Interpolator->EvaluateAtContinuousIndex(cdx)));
//    }
//  else return ( static_cast<ScalarValueType>(m_SpeedImage->GetPixel(idx)) );
//}
//
//template <class TImageType, class TFeatureImageType>
//typename SegmentationLevelSetFunction<TImageType, TFeatureImageType>::VectorType
//SegmentationLevelSetFunction<TImageType, TFeatureImageType>
//::AdvectionField(const NeighborhoodType &neighborhood,
//                 const FloatOffsetType &offset, GlobalDataStruct *)  const
//{
//  IndexType idx = neighborhood.GetIndex();
//  ContinuousIndexType cdx;
//  for (unsigned i = 0; i < ImageDimension; ++i)
//    {
//    cdx[i] = static_cast<double>(idx[i]) - offset[i];
//    }
//  if ( m_VectorInterpolator->IsInsideBuffer(cdx) )
//    {
//    return ( m_VectorCast(m_VectorInterpolator->EvaluateAtContinuousIndex(cdx)));
//    }
//  //Just return the default else
//    return ( m_AdvectionImage->GetPixel(idx) );
//  
//}
//
/////////////////////////////////////////////////////////////
//// ACTUAL computation
//
//template< class TImageType >
//typename LevelSetFunction< TImageType >::PixelType
//LevelSetFunction< TImageType >
//::ComputeUpdate(const NeighborhoodType &it, void *globalData,
//                const FloatOffsetType& offset)
//{
//  unsigned int i, j;  
//  const ScalarValueType ZERO = NumericTraits<ScalarValueType>::Zero;
//  const ScalarValueType center_value  = it.GetCenterPixel();
//
//  const NeighborhoodScalesType neighborhoodScales = this->ComputeNeighborhoodScales();
//
//  ScalarValueType laplacian, x_energy, laplacian_term, propagation_term,
//    curvature_term, advection_term, propagation_gradient;
//  VectorType advection_field;
//
//  // Global data structure
//  GlobalDataStruct *gd = (GlobalDataStruct *)globalData;
//
//  // Compute the Hessian matrix and various other derivatives.  Some of these
//  // derivatives may be used by overloaded virtual functions.
//  gd->m_GradMagSqr = 1.0e-6;
//  for( i = 0 ; i < ImageDimension; i++)
//    {
//    const unsigned int positionA = 
//      static_cast<unsigned int>( m_Center + m_xStride[i]);    
//    const unsigned int positionB = 
//      static_cast<unsigned int>( m_Center - m_xStride[i]);    
//
//    gd->m_dx[i] = 0.5 * (it.GetPixel( positionA ) - 
//                         it.GetPixel( positionB ) ) * neighborhoodScales[i]; 
//    gd->m_dxy[i][i] = ( it.GetPixel( positionA )
//                      + it.GetPixel( positionB ) - 2.0 * center_value ) *
//                                            vnl_math_sqr(neighborhoodScales[i]) ;
//
//    gd->m_dx_forward[i]  = ( it.GetPixel( positionA ) - center_value ) * neighborhoodScales[i];
//    gd->m_dx_backward[i] = ( center_value - it.GetPixel( positionB ) ) * neighborhoodScales[i];
//    gd->m_GradMagSqr += gd->m_dx[i] * gd->m_dx[i];
//
//    for( j = i+1; j < ImageDimension; j++ )
//      {
//      const unsigned int positionAa = static_cast<unsigned int>( 
//        m_Center - m_xStride[i] - m_xStride[j] );
//      const unsigned int positionBa = static_cast<unsigned int>( 
//        m_Center - m_xStride[i] + m_xStride[j] );
//      const unsigned int positionCa = static_cast<unsigned int>( 
//        m_Center + m_xStride[i] - m_xStride[j] );
//      const unsigned int positionDa = static_cast<unsigned int>( 
//        m_Center + m_xStride[i] + m_xStride[j] );
//
//      gd->m_dxy[i][j] = gd->m_dxy[j][i] = 0.25 * ( it.GetPixel( positionAa )
//                                                 - it.GetPixel( positionBa )
//                                                 - it.GetPixel( positionCa )
//                                                 + it.GetPixel( positionDa ) )
//                                          * neighborhoodScales[i] * neighborhoodScales[j] ;
//      }
//    }
//
//  if ( m_CurvatureWeight != ZERO )
//    {
//    curvature_term = this->ComputeCurvatureTerm(it, offset, gd) * m_CurvatureWeight
//      * this->CurvatureSpeed(it, offset);
//
//    gd->m_MaxCurvatureChange = vnl_math_max(gd->m_MaxCurvatureChange,
//                   vnl_math_abs(curvature_term));
//    }
//  else
//    {
//    curvature_term = ZERO;
//    }
//
//  // Calculate the advection term.
//  //  $\alpha \stackrel{\rightharpoonup}{F}(\mathbf{x})\cdot\nabla\phi $
//  //
//  // Here we can use a simple upwinding scheme since we know the
//  // sign of each directional component of the advective force.
//  //
//  if (m_AdvectionWeight != ZERO)
//    {
//    
//    advection_field = this->AdvectionField(it, offset, gd);
//    advection_term = ZERO;
//    
//    for(i = 0; i < ImageDimension; i++)
//      {
//      
//      x_energy = m_AdvectionWeight * advection_field[i];
//      
//      if (x_energy > ZERO)
//        {
//        advection_term += advection_field[i] * gd->m_dx_backward[i];
//        }
//      else
//        {
//        advection_term += advection_field[i] * gd->m_dx_forward[i];
//        }
//        
//      gd->m_MaxAdvectionChange
//        = vnl_math_max(gd->m_MaxAdvectionChange, vnl_math_abs(x_energy)); 
//      }
//    advection_term *= m_AdvectionWeight;
//    
//    }
//  else
//    {
//    advection_term = ZERO;
//    }
//
//  if (m_PropagationWeight != ZERO)
//    {
//    // Get the propagation speed
//    propagation_term = m_PropagationWeight * this->PropagationSpeed(it, offset, gd);
//      
//    //
//    // Construct upwind gradient values for use in the propagation speed term:
//    //  $\beta G(\mathbf{x})\mid\nabla\phi\mid$
//    //
//    // The following scheme for ``upwinding'' in the normal direction is taken
//    // from Sethian, Ch. 6 as referenced above.
//    //
//    propagation_gradient = ZERO;
//    
//    if ( propagation_term > ZERO )
//      {
//      for(i = 0; i< ImageDimension; i++)
//        {
//        propagation_gradient += vnl_math_sqr( vnl_math_max(gd->m_dx_backward[i], ZERO) )
//          + vnl_math_sqr( vnl_math_min(gd->m_dx_forward[i],  ZERO) );
//        }
//      }
//    else
//      {
//      for(i = 0; i< ImageDimension; i++)
//        {
//        propagation_gradient += vnl_math_sqr( vnl_math_min(gd->m_dx_backward[i], ZERO) )
//          + vnl_math_sqr( vnl_math_max(gd->m_dx_forward[i],  ZERO) );
//        }        
//      }
//    
//    // Collect energy change from propagation term.  This will be used in
//    // calculating the maximum time step that can be taken for this iteration.
//    gd->m_MaxPropagationChange =
//      vnl_math_max(gd->m_MaxPropagationChange,
//                   vnl_math_abs(propagation_term));
//    
//    propagation_term *= vcl_sqrt( propagation_gradient );
//    }
//  else propagation_term = ZERO;
//
//  if(m_LaplacianSmoothingWeight != ZERO)
//    {
//    laplacian = ZERO;
//    
//    // Compute the laplacian using the existing second derivative values
//    for(i = 0;i < ImageDimension; i++)
//      {
//      laplacian += gd->m_dxy[i][i];
//      }
//
//    // Scale the laplacian by its speed and weight
//    laplacian_term = 
//      laplacian * m_LaplacianSmoothingWeight * LaplacianSmoothingSpeed(it,offset, gd);
//    }
//  else 
//    laplacian_term = ZERO;
//
//  // Return the combination of all the terms.
//  return ( PixelType ) ( curvature_term - propagation_term 
//                         - advection_term - laplacian_term );
//} 

/////////////////////

}
#endif