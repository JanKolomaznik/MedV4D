#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#error File diffFunc.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template <class TInputNeighbour, class TFeatureNeighbour>
ThresholdLevelSetFunc< TInputNeighbour, TFeatureNeighbour >
::ThresholdLevelSetFunc()
{
	m_WaveDT = 1.0/(2.0 * TInputNeighbour::Dim);
	m_DT = 1.0/(2.0 * TInputNeighbour::Dim);
	
	RadiusType radius(1,1,1);
	this->SetRadius(radius);
	  
//  // Dummy neighborhood.
//  NeighborhoodType it;
//  it.SetRadius( radius );
//  
//  // Find the center index of the neighborhood.
//  m_Center =  it.Size() / 2;
//
//  // Get the stride length for each axis.
//  for(unsigned int i = 0; i < TInputNeighbour::Dim; i++)
//    {  m_xStride[i] = it.GetStride(i); }
  
  // initialize variables
  for (unsigned int i = 0; i < TInputNeighbour::Dim; i++)
    {
    m_ScaleCoefficients[i] = 1.0;
    }
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
	
template <class TInputNeighbour, class TFeatureNeighbour>
typename ThresholdLevelSetFunc< TInputNeighbour, TFeatureNeighbour >::PixelType
ThresholdLevelSetFunc< TInputNeighbour, TFeatureNeighbour >
::ComputeUpdate(const NeighborhoodType &it, const TFeatureNeighbour &featureNeib, void *globalData,
		const FloatOffsetType& offset)
{
	unsigned int i, j;
	//const PixelType ZERO = NumericTraits<PixelType>::Zero;
	const PixelType center_value = it.GetCenterPixel();

	const NeighborhoodScalesType neighborhoodScales = this->ComputeNeighborhoodScales();

	// Global data structure
	GlobalDataType *gd = (GlobalDataType *)globalData;
	
	typename TInputNeighbour::StrideType stride = it.GetNeighborhood().GetStrides();
	uint32 m_Center = it.GetNeighborhood().GetSize() / 2;

	// Compute the Hessian matrix and various other derivatives.  Some of these
	// derivatives may be used by overloaded virtual functions.
	gd->m_GradMagSqr = 1.0e-6;
	for( i = 0; i < TInputNeighbour::Dim; i++)
	{
		const unsigned int positionA =
		static_cast<unsigned int>( m_Center + stride[i] );
		const unsigned int positionB =
		static_cast<unsigned int>( m_Center - stride[i] );

		gd->m_dx[i] = 0.5 * (it.GetPixel( positionA ) -
				it.GetPixel( positionB ) ) * neighborhoodScales[i];
		gd->m_dxy[i][i] = ( it.GetPixel( positionA )
				+ it.GetPixel( positionB ) - 2.0 * center_value ) *
		vnl_math_sqr(neighborhoodScales[i]);

		gd->m_dx_forward[i] = ( it.GetPixel( positionA ) - center_value ) * neighborhoodScales[i];
		gd->m_dx_backward[i] = ( center_value - it.GetPixel( positionB ) ) * neighborhoodScales[i];
		gd->m_GradMagSqr += gd->m_dx[i] * gd->m_dx[i];

		for( j = i+1; j < TInputNeighbour::Dim; j++ )
		{
			const unsigned int positionAa = static_cast<unsigned int>(
					m_Center - stride[i] - stride[j] );
			const unsigned int positionBa = static_cast<unsigned int>(
					m_Center - stride[i] + stride[j] );
			const unsigned int positionCa = static_cast<unsigned int>(
					m_Center + stride[i] - stride[j] );
			const unsigned int positionDa = static_cast<unsigned int>(
					m_Center + stride[i] + stride[j] );

			gd->m_dxy[i][j] = gd->m_dxy[j][i] = 0.25 * ( it.GetPixel( positionAa )
					- it.GetPixel( positionBa )
					- it.GetPixel( positionCa )
					+ it.GetPixel( positionDa ) )
			* neighborhoodScales[i] * neighborhoodScales[j];
		}
	}

	// Return the combination of all the terms.
	PixelType result = ( PixelType )( 
			this->ComputeCurvatureTerm(gd) - 
			this->ComputePropagationTerm(featureNeib, offset, gd)
			//- ComputeAdvectionTerm()
			);
	
	return result;
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputNeighbour, class TFeatureNeighbour>
typename ThresholdLevelSetFunc< TInputNeighbour, TFeatureNeighbour >::TimeStepType
ThresholdLevelSetFunc< TInputNeighbour, TFeatureNeighbour >
	::ComputeGlobalTimeStep(void *GlobalData) const
{
  TimeStepType dt;

  GlobalDataType *d = (GlobalDataType *)GlobalData;

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
  for (unsigned int i=0; i<TInputNeighbour::Dim; i++)
    {
    maxScaleCoefficient = vnl_math_max( (double)this->m_ScaleCoefficients[i],maxScaleCoefficient);
    }
  dt /= maxScaleCoefficient;
 
  // reset the values  
  d->m_MaxAdvectionChange   = itk::NumericTraits<PixelType>::Zero;
  d->m_MaxPropagationChange = itk::NumericTraits<PixelType>::Zero;
  d->m_MaxCurvatureChange   = itk::NumericTraits<PixelType>::Zero;
  
  return dt;
}

///////////////////////////////////////////////////////////////////////////////

#endif
