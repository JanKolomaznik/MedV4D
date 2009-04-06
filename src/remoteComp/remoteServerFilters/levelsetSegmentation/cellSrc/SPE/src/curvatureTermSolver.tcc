#ifndef CURVATURETERMSOLVER_H_
#error File curvatureTermSolver.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template< typename PixelType, uint8 Dim >
typename CurvatureTermSolver< PixelType, Dim>::ScalarValueType
CurvatureTermSolver< PixelType, Dim>
::ComputeCurvatureTerm(GlobalDataType *gd)
{
	if ( m_CurvatureWeight == 0)//itk::NumericTraits<ScalarValueType>::Zero )
		return 0;//itk::NumericTraits<ScalarValueType>::Zero;

	ScalarValueType curvature_term =
	this->ComputeMeanCurvature(gd) * m_CurvatureWeight;
		//* this->CurvatureSpeed(it, offset);

	gd->m_MaxCurvatureChange = vnl_math_max(gd->m_MaxCurvatureChange,
			vnl_math_abs(curvature_term));
	
	return curvature_term;
}

///////////////////////////////////////////////////////////////////////////////

//template< class TImageType>
//typename LevelSetFunction< TImageType >::ScalarValueType
//LevelSetFunction< TImageType >
//::ComputeMinimalCurvature(
//  const NeighborhoodType &itkNotUsed(neighborhood),
//  const FloatOffsetType& itkNotUsed(offset), GlobalDataStruct *gd)
//{
//
//  unsigned int i, j, n;
//  ScalarValueType gradMag = vcl_sqrt(gd->m_GradMagSqr);
//  ScalarValueType Pgrad[ImageDimension][ImageDimension];
//  ScalarValueType tmp_matrix[ImageDimension][ImageDimension];
//  const ScalarValueType ZERO = NumericTraits<ScalarValueType>::Zero;
//  vnl_matrix_fixed<ScalarValueType, ImageDimension, ImageDimension> Curve;
//  const ScalarValueType MIN_EIG = NumericTraits<ScalarValueType>::min();
//
//  ScalarValueType mincurve; 
//
//  for (i = 0; i < ImageDimension; i++)
//    {
//    Pgrad[i][i] = 1.0 - gd->m_dx[i] * gd->m_dx[i]/gradMag;
//    for (j = i+1; j < ImageDimension; j++)
//      {
//      Pgrad[i][j]= gd->m_dx[i] * gd->m_dx[j]/gradMag;
//      Pgrad[j][i] = Pgrad[i][j];
//      }
//    }
//
//  //Compute Pgrad * Hessian * Pgrad
//  for (i = 0; i < ImageDimension; i++)
//    {
//    for (j = i; j < ImageDimension; j++)
//      {
//      tmp_matrix[i][j]= ZERO;
//      for (n = 0 ; n < ImageDimension; n++)
//        {
//        tmp_matrix[i][j] += Pgrad[i][n] * gd->m_dxy[n][j];
//        }
//      tmp_matrix[j][i]=tmp_matrix[i][j];
//      }
//    }
//
//  for (i = 0; i < ImageDimension; i++)
//    {
//    for (j = i; j < ImageDimension; j++)
//      {
//      Curve(i,j) = ZERO;
//      for (n = 0 ; n < ImageDimension; n++)
//        {
//        Curve(i,j) += tmp_matrix[i][n] * Pgrad[n][j];
//        }
//      Curve(j,i) = Curve(i,j);
//      }
//    }
//
//  //Eigensystem
//  vnl_symmetric_eigensystem<ScalarValueType>  eig(Curve);
//
//  mincurve=vnl_math_abs(eig.get_eigenvalue(ImageDimension-1));
//  for (i = 0; i < ImageDimension; i++)
//    {
//    if(vnl_math_abs(eig.get_eigenvalue(i)) < mincurve &&
//       vnl_math_abs(eig.get_eigenvalue(i)) > MIN_EIG)
//      {
//      mincurve = vnl_math_abs(eig.get_eigenvalue(i));
//      }
//    }
//
//  return ( mincurve / gradMag );  
//}

///////////////////////////////////////////////////////////////////////////////

//template< class TImageType>
//typename LevelSetFunction< TImageType >::ScalarValueType
//LevelSetFunction< TImageType >
//::Compute3DMinimalCurvature(const NeighborhoodType &neighborhood,
//                            const FloatOffsetType& offset, GlobalDataStruct *gd)
//{
//  ScalarValueType mean_curve = this->ComputeMeanCurvature(neighborhood, offset, gd);
//  
//  int i0 = 0, i1 = 1, i2 = 2;
//  ScalarValueType gauss_curve =
//    (2*(gd->m_dx[i0]*gd->m_dx[i1]*(gd->m_dxy[i2][i0]
//                                   *gd->m_dxy[i1][i2]-gd->m_dxy[i0][i1]*gd->m_dxy[i2][i2]) +
//        gd->m_dx[i1]*gd->m_dx[i2]*(gd->m_dxy[i2][i0]
//                                   *gd->m_dxy[i0][i1]-gd->m_dxy[i1][i2]*gd->m_dxy[i0][i0]) +
//        gd->m_dx[i0]*gd->m_dx[i2]*(gd->m_dxy[i1][i2]
//                                   *gd->m_dxy[i0][i1]-gd->m_dxy[i2][i0]*gd->m_dxy[i1][i1])) +
//     gd->m_dx[i0]*gd->m_dx[i0]*(gd->m_dxy[i1][i1]
//                                *gd->m_dxy[i2][i2]-gd->m_dxy[i1][i2]*gd->m_dxy[i1][i2]) +
//     gd->m_dx[i1]*gd->m_dx[i1]*(gd->m_dxy[i0][i0]
//                                *gd->m_dxy[i2][i2]-gd->m_dxy[i2][i0]*gd->m_dxy[i2][i0]) +
//     gd->m_dx[i2]*gd->m_dx[i2]*(gd->m_dxy[i1][i1]
//                                *gd->m_dxy[i0][i0]-gd->m_dxy[i0][i1]*gd->m_dxy[i0][i1]))/
//    (gd->m_dx[i0]*gd->m_dx[i0] + gd->m_dx[i1]*gd->m_dx[i1] + gd->m_dx[i2]*gd->m_dx[i2]);
//  
//  ScalarValueType discriminant = mean_curve * mean_curve-gauss_curve;
//  if (discriminant < 0.0)
//    {
//    discriminant = 0.0;
//    }
//  discriminant = vcl_sqrt(discriminant);
//  return  (mean_curve - discriminant);
//}

///////////////////////////////////////////////////////////////////////////////

template< typename PixelType, uint8 Dim >  
typename CurvatureTermSolver< PixelType, Dim>::ScalarValueType
CurvatureTermSolver< PixelType, Dim>
	::ComputeMeanCurvature(GlobalDataType *gd)
{
  // Calculate the mean curvature
  ScalarValueType curvature_term = 0;//itk::NumericTraits<ScalarValueType>::Zero;
  unsigned int i, j;
  
  for (i = 0; i < Dim; i++)
    {      
    for(j = 0; j < Dim; j++)
      {      
      if(j != i)
        {
        curvature_term -= gd->m_dx[i] * gd->m_dx[j] * gd->m_dxy[i][j];
        curvature_term += gd->m_dxy[j][j] * gd->m_dx[i] * gd->m_dx[i];
        }
      }
    }
  
  return (curvature_term / gd->m_GradMagSqr );
}

///////////////////////////////////////////////////////////////////////////////


#endif
