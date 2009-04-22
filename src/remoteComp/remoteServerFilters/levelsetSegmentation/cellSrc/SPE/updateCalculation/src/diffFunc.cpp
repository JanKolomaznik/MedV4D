#include "common/Types.h"
#include "../diffFunc.h"
#include "../../vnl_math.h"
#include "common/Debug.h"

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////

ThresholdLevelSetFunc::ThresholdLevelSetFunc()
{
	m_WaveDT = 1.0/(2.0 * DIM);
	m_DT = 1.0/(2.0 * DIM);

	// initialize variables
	for (unsigned int i = 0; i < DIM; i++)
	{
		m_ScaleCoefficients[i] = 1.0;
	}
	
	for(int i=0; i<DIM; i++)
  {
		_neighborhoodScales[i] = this->m_ScaleCoefficients[i] / RADIUS;
  }
}

///////////////////////////////////////////////////////////////////////////////


TPixelValue ThresholdLevelSetFunc::ComputeUpdate(
		const NeighborhoodIteratorType &it,
		const NeighborhoodIteratorType &featureNeib,
		GlobalDataStruct *globalData, const TContinuousIndex& offset)
{
	unsigned int i, j;
	//const PixelType ZERO = NumericTraits<PixelType>::Zero;
	const TPixelValue center_value = it.GetCenterPixel();

	TStrides stride = it.GetNeighborhood().GetStrides();
	uint32 m_Center = it.GetNeighborhood().GetSize() / 2;
	
	//D_PRINT( it.GetNeighborhood() );

	// Compute the Hessian matrix and various other derivatives.  Some of these
	// derivatives may be used by overloaded virtual functions.
	globalData->m_GradMagSqr = 1.0e-6;
	for (i = 0; i < DIM; i++)
	{
		const unsigned int positionA = static_cast<unsigned int>(m_Center
				+ stride[i] );
		const unsigned int positionB = static_cast<unsigned int>(m_Center
				- stride[i] );

		globalData->m_dx[i] = 0.5 * (it.GetPixel(positionA)
				- it.GetPixel(positionB) ) * _neighborhoodScales[i];
		globalData->m_dxy[i][i] = (it.GetPixel(positionA)
				+ it.GetPixel(positionB) - 2.0 * center_value )
				* vnl_math_sqr(_neighborhoodScales[i]);

		globalData->m_dx_forward[i] = (it.GetPixel(positionA) - center_value )
				* _neighborhoodScales[i];
		globalData->m_dx_backward[i] = (center_value - it.GetPixel(positionB) )
				* _neighborhoodScales[i];
		globalData->m_GradMagSqr += globalData->m_dx[i] * globalData->m_dx[i];

		for (j = i+1; j < DIM; j++)
		{
			const unsigned int positionAa = static_cast<unsigned int>(m_Center
					- stride[i] - stride[j] );
			const unsigned int positionBa = static_cast<unsigned int>(m_Center
					- stride[i] + stride[j] );
			const unsigned int positionCa = static_cast<unsigned int>(m_Center
					+ stride[i] - stride[j] );
			const unsigned int positionDa = static_cast<unsigned int>(m_Center
					+ stride[i] + stride[j] );

			globalData->m_dxy[i][j] = globalData->m_dxy[j][i]
					= 0.25
							* (it.GetPixel(positionAa)
									- it.GetPixel(positionBa)
									- it.GetPixel(positionCa)
									+ it.GetPixel(positionDa) )
							* _neighborhoodScales[i] * _neighborhoodScales[j];
		}
	}

	// Return the combination of all the terms.
	TPixelValue result = ( TPixelValue )(this->ComputeCurvatureTerm(globalData)
			- this->ComputePropagationTerm(featureNeib, offset, globalData)
			//- ComputeAdvectionTerm()
	);

	return result;
}

///////////////////////////////////////////////////////////////////////////////

TimeStepType ThresholdLevelSetFunc::ComputeGlobalTimeStep(void *GlobalData)
{
	TimeStepType dt;

	GlobalDataType *d = (GlobalDataType *)GlobalData;

	d->m_MaxAdvectionChange += d->m_MaxPropagationChange;

	if (vnl_math_abs(d->m_MaxCurvatureChange) > 0.0)
	{
		if (d->m_MaxAdvectionChange > 0.0)
		{
			dt = vnl_math_min((m_WaveDT / d->m_MaxAdvectionChange), (m_DT
					/ d->m_MaxCurvatureChange ));
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
	for (unsigned int i=0; i<DIM; i++)
	{
		maxScaleCoefficient = vnl_math_max(
				(double)this->m_ScaleCoefficients[i], maxScaleCoefficient);
	}
	dt /= maxScaleCoefficient;

	// reset the values  
	d->m_MaxAdvectionChange = 0;
	d->m_MaxPropagationChange = 0;
	d->m_MaxCurvatureChange = 0;

	return dt;
}

///////////////////////////////////////////////////////////////////////////////
