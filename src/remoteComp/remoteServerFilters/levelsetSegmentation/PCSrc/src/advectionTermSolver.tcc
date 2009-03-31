#ifndef ADVECTIONTERMSOLVER_H_
#error File advectionTermSolver.tcccannot be included directly!
#else

namespace itk
{

///////////////////////////////////////////////////////////////////////////////

template <class TImageType, typename NeighborhoodType, typename PixelType, typename FloatOffsetType>
typename ThresholdLevelSetFunc<TImageType, TFeatureImageType>::PixelType
ThresholdLevelSetFunc<TImageType, TFeatureImageType>
::ComputeAdvectionTerm(void)
{
	if (m_AdvectionWeight == ZERO)
		return ZERO;

	// Calculate the advection term.
	//  $\alpha \stackrel{\rightharpoonup}{F}(\mathbf{x})\cdot\nabla\phi $
	//
	// Here we can use a simple upwinding scheme since we know the
	// sign of each directional component of the advective force.
	//
	VectorType advection_field = this->AdvectionField(it, offset, gd);
	ScalarValueType advection_term = ZERO;
	ScalarValueType x_energy;

	for(i = 0; i < ImageDimension; i++)
	{
		x_energy = m_AdvectionWeight * advection_field[i];

		if (x_energy> ZERO)
		{
			advection_term += advection_field[i] * gd->m_dx_backward[i];
		}
		else
		{
			advection_term += advection_field[i] * gd->m_dx_forward[i];
		}

		gd->m_MaxAdvectionChange
		= vnl_math_max(gd->m_MaxAdvectionChange, vnl_math_abs(x_energy));
	}
	advection_term *= m_AdvectionWeight;

	return advection_term;
}

///////////////////////////////////////////////////////////////////////////////

template <class TImageType, class TFeatureImageType>
typename SegmentationLevelSetFunction<TImageType, TFeatureImageType>::VectorType
SegmentationLevelSetFunction<TImageType, TFeatureImageType>
::AdvectionField(const NeighborhoodType &neighborhood,
                 const FloatOffsetType &offset, GlobalDataStruct *)  const
{
  IndexType idx = neighborhood.GetIndex();
  ContinuousIndexType cdx;
  for (unsigned i = 0; i < ImageDimension; ++i)
    {
    cdx[i] = static_cast<double>(idx[i]) - offset[i];
    }
  if ( m_VectorInterpolator->IsInsideBuffer(cdx) )
    {
    return ( m_VectorCast(m_VectorInterpolator->EvaluateAtContinuousIndex(cdx)));
    }
  //Just return the default else
  return ( m_AdvectionImage->GetPixel(idx) );
  
}

///////////////////////////////////////////////////////////////////////////////

}

#endif