
#include "common/Types.h"
#include "common/Debug.h"
#include "../updateCalculatorSPE.h"
#include "../../vnl_math.h"
#include <string.h>

using namespace M4D::Cell;


///////////////////////////////////////////////////////////////////////////////

void 
M4D::Cell::ComputeStridesFromSize(const TSize &size, TStrides &strides)
{
  unsigned int accum;

  accum = 1;
  strides[0] = 1;
  for (unsigned int dim = 1; dim < DIM; ++dim)
    {
	  accum *= size[dim-1];
	  strides[dim] = accum;
	  }
}

TIndex M4D::Cell::operator+(const TIndex &i, const TOffset &o)
{
	TIndex ret;
	ret[0] = i[0] + o[0];
	ret[1] = i[1] + o[1];
	ret[2] = i[2] + o[2];
	return ret;
}

///////////////////////////////////////////////////////////////////////////////

UpdateCalculatorSPE
::UpdateCalculatorSPE()
{
	memset(&m_globalData, 0, sizeof(GlobalDataStruct));
}

///////////////////////////////////////////////////////////////////////////////


void
UpdateCalculatorSPE
::Init(void)
{
	MIN_NORM = 1.0e-6;

	double minSpacing = 1000000;//itk::NumericTraits<double>::max();
	for (uint8 i=0; i<DIM; i++)
	{
		minSpacing = vnl_math_min(minSpacing, (double) m_Conf->valueImageProps.spacing[i]);
	}
	MIN_NORM *= minSpacing;

	// set props to diffFunc
	m_diffFunc.SetUpperThreshold(m_Conf->m_upThreshold);
	m_diffFunc.SetLowerThreshold(m_Conf->m_downThreshold);
	m_diffFunc.SetPropagationWeight(m_Conf->m_propWeight);
	m_diffFunc.SetCurvatureWeight(m_Conf->m_curvWeight);
}

///////////////////////////////////////////////////////////////////////////////

void
UpdateCalculatorSPE
::CalculateChangeItem(void)
{		
	// Calculate the offset to the surface from the center of this
	// neighborhood.  This is used by some level set functions in sampling a
	// speed, advection, or curvature term.
	if ((centerValue = m_outIter.GetCenterPixel()) != 0.0 )
	{
		// Surface is at the zero crossing, so distance to surface is:
		// phi(x) / norm(grad(phi)), where phi(x) is the center of the
		// neighborhood.  The location is therefore
		// (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
		norm_grad_phi_squared = 0.0;
		for (i = 0; i < DIM; ++i)
		{
			forwardValue = m_outIter.GetNext(i);
			backwardValue = m_outIter.GetPrevious(i);

			if (forwardValue * backwardValue >= 0)
			{ //  Neighbors are same sign OR at least one neighbor is zero.
				dx_forward = forwardValue - centerValue;
				dx_backward = centerValue - backwardValue;

				// Pick the larger magnitude derivative.
				if (vnl_math_abs(dx_forward)> vnl_math_abs(dx_backward) )
				{
					offset[i] = dx_forward;
				}
				else
				{
					offset[i] = dx_backward;
				}
			}
			else //Neighbors are opposite sign, pick the direction of the 0 surface.
			{
				if (forwardValue * centerValue < 0)
				{
					offset[i] = forwardValue - centerValue;
				}
				else
				{
					offset[i] = centerValue - backwardValue;
				}
			}

			norm_grad_phi_squared += offset[i] * offset[i];
		}

		for (i = 0; i < DIM; ++i)
		{
			offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
		}

		m_updateBufferArray.push_back( 
				m_diffFunc.ComputeUpdate(m_outIter, m_featureIter, &m_globalData, offset) );
	}
}

///////////////////////////////////////////////////////////////////////////////

TimeStepType
UpdateCalculatorSPE::CalculateChange()
{
	m_updateBufferArray.SetArray(m_Conf->m_UpdateBufferData);
	m_layerIterator.SetBeginEnd(m_Conf->m_activeSetBegin, m_Conf->m_activeSetEnd);
	
	// prepare neighbour preloaders
	m_valueNeighbPreloader.SetImageProps(& m_Conf->valueImageProps);
	m_featureNeighbPreloader.SetImageProps(& m_Conf->featureImageProps);
	
	// Calculates the update values for the active layer indicies in this
	// iteration.  Iterates through the active layer index list, applying 
	// the level set function to the output image (level set image) at each
	// index.  Update values are stored in the update buffer.
	LayerNodeType *next;
	while(m_layerIterator.HasNext())
	{
		next = m_layerIterator.Next();
		
		// load approp neigborhood
		m_valueNeighbPreloader.Load(next->m_Value);
		m_featureNeighbPreloader.Load(next->m_Value);
		
		m_outIter.SetNeighbourhood( m_valueNeighbPreloader.GetLoaded());
		m_featureIter.SetNeighbourhood( m_featureNeighbPreloader.GetLoaded());
				
		CalculateChangeItem();
	}
	
	m_updateBufferArray.FlushArray();

	// Ask the finite difference function to compute the time step for
	// this iteration.  We give it the global data pointer to use, then
	// ask it to free the global data memory.
	TimeStepType timeStep = m_diffFunc.ComputeGlobalTimeStep(&m_globalData);

	return timeStep;
}

///////////////////////////////////////////////////////////////////////////////

