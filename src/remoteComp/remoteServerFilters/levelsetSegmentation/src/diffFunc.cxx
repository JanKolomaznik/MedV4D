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
	
	this->SetRadius(r);
	  
  // Dummy neighborhood.
  NeighborhoodType it;
  it.SetRadius( r );
  
  // Find the center index of the neighborhood.
  m_Center =  it.Size() / 2;

  // Get the stride length for each axis.
  for(unsigned int i = 0; i < ImageDimension; i++)
    {  m_xStride[i] = it.GetStride(i); }

	cntr_.Reset();
}

///////////////////////////////////////////////////////////////////////////////

	//template <class TImageType, class TFeatureImageType>
	//typename ThresholdLevelSetFunc<TImageType, TFeatureImageType>::PixelType
	//ThresholdLevelSetFunc<TImageType, TFeatureImageType>
	//	::ComputeUpdate(const NeighborhoodType &neighborhood, void *globalData,
	//            const FloatOffsetType& offset)
	//{
	//	typedef LevelSetFunction<TImageType> LSFunc;
	//	typedef typename LSFunc::NeighborhoodType LSNeighborhoodType;
	//	typedef typename LSFunc::FloatOffsetType LSFloatOffsetType;
	//	
	//	cntr_.Start();
	//	PixelType retval = LevelSetFunction<TImageType>::ComputeUpdate(
	//			(const LSNeighborhoodType &) neighborhood,
	//			globalData,
	//			(const LSFloatOffsetType &) offset);
	//	cntr_.Stop();
	//	
	//	return retval;
	//}
	
///////////////////////////////////////////////////////////////////////////////
	
template< class TImageType>
typename LevelSetFunction< TImageType>::PixelType
LevelSetFunction< TImageType>
::ComputeUpdate(const NeighborhoodType &it, void *globalData,
		const FloatOffsetType& offset)
{
	unsigned int i, j;
	const ScalarValueType ZERO = NumericTraits<ScalarValueType>::Zero;
	const ScalarValueType center_value = it.GetCenterPixel();

	const NeighborhoodScalesType neighborhoodScales = this->ComputeNeighborhoodScales();

	// Global data structure
	GlobalDataStruct *gd = (GlobalDataStruct *)globalData;

	// Compute the Hessian matrix and various other derivatives.  Some of these
	// derivatives may be used by overloaded virtual functions.
	gd->m_GradMagSqr = 1.0e-6;
	for( i = 0; i < ImageDimension; i++)
	{
		const unsigned int positionA =
		static_cast<unsigned int>( m_Center + m_xStride[i]);
		const unsigned int positionB =
		static_cast<unsigned int>( m_Center - m_xStride[i]);

		gd->m_dx[i] = 0.5 * (it.GetPixel( positionA ) -
				it.GetPixel( positionB ) ) * neighborhoodScales[i];
		gd->m_dxy[i][i] = ( it.GetPixel( positionA )
				+ it.GetPixel( positionB ) - 2.0 * center_value ) *
		vnl_math_sqr(neighborhoodScales[i]);

		gd->m_dx_forward[i] = ( it.GetPixel( positionA ) - center_value ) * neighborhoodScales[i];
		gd->m_dx_backward[i] = ( center_value - it.GetPixel( positionB ) ) * neighborhoodScales[i];
		gd->m_GradMagSqr += gd->m_dx[i] * gd->m_dx[i];

		for( j = i+1; j < ImageDimension; j++ )
		{
			const unsigned int positionAa = static_cast<unsigned int>(
					m_Center - m_xStride[i] - m_xStride[j] );
			const unsigned int positionBa = static_cast<unsigned int>(
					m_Center - m_xStride[i] + m_xStride[j] );
			const unsigned int positionCa = static_cast<unsigned int>(
					m_Center + m_xStride[i] - m_xStride[j] );
			const unsigned int positionDa = static_cast<unsigned int>(
					m_Center + m_xStride[i] + m_xStride[j] );

			gd->m_dxy[i][j] = gd->m_dxy[j][i] = 0.25 * ( it.GetPixel( positionAa )
					- it.GetPixel( positionBa )
					- it.GetPixel( positionCa )
					+ it.GetPixel( positionDa ) )
			* neighborhoodScales[i] * neighborhoodScales[j];
		}
	}

	// Return the combination of all the terms.
	return ( PixelType )( 
			ComputeCurvatureTerm() - 
			ComputePropagationTerm() - 
			ComputeAdvectionTerm() 
			);
}

///////////////////////////////////////////////////////////////////////////////

template< class TImageType >
typename ThresholdLevelSetFunc< TImageType >::TimeStepType
ThresholdLevelSetFunc<TImageType>
	::ComputeGlobalTimeStep(void *GlobalData) const
{
  TimeStepType dt;

  GlobalDataStruct *d = (GlobalDataStruct *)GlobalData;

  d->m_MaxAdvectionChange += d->m_MaxPropagationChange;
  
  if (vnl_math_abs(d->m_MaxCurvatureChange) > 0.0)
    {
    if (d->m_MaxAdvectionChange > 0.0)
      {
      dt = vnl_math_min((m_WaveDT / d->m_MaxAdvectionChange),
                        (    m_DT / d->m_MaxCurvatureChange ));
      }
    else
      {
      dt = m_DT / d->m_MaxCurvatureChange;
      }
    }
  else
    {
    if (d->m_MaxAdvectionChange > 0.0)
      {
      dt = m_WaveDT / d->m_MaxAdvectionChange; 
      }
    else 
      {
      dt = 0.0;
      }
    }

  double maxScaleCoefficient = 0.0;
  for (unsigned int i=0; i<ImageDimension; i++)
    {
    maxScaleCoefficient = vnl_math_max(this->m_ScaleCoefficients[i],maxScaleCoefficient);
    }
  dt /= maxScaleCoefficient;
 
  // reset the values  
  d->m_MaxAdvectionChange   = NumericTraits<ScalarValueType>::Zero;
  d->m_MaxPropagationChange = NumericTraits<ScalarValueType>::Zero;
  d->m_MaxCurvatureChange   = NumericTraits<ScalarValueType>::Zero;
  
  return dt;
}

///////////////////////////////////////////////////////////////////////////////

}
#endif
