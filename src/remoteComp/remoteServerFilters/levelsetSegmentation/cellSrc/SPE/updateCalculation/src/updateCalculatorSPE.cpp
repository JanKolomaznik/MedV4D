
#ifdef FOR_CELL
#include "../../tools/SPEdebug.h"
#else
#include "common/Debug.h"
#endif

#include "common/Types.h"
#include "../updateCalculatorSPE.h"
#include "../../vnl_math.h"
#include <string.h>

#define UPDATE_CALC_DEBUG 12
#define DBG_LAYER_IT 12

using namespace M4D::Cell;


///////////////////////////////////////////////////////////////////////////////

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
::UpdateCalculatorSPE(SharedResources *shaRes)
	: m_updateBufferArray(shaRes->_buf)
	, m_Conf(&shaRes->_runConf)
	, m_stepConfig(&shaRes->_changeConfig)
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

	UpdateFunctionProperties();
}

///////////////////////////////////////////////////////////////////////////////

void UpdateCalculatorSPE::UpdateFunctionProperties()
{
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
#ifndef FOR_CELL
	DL_PRINT(DBG_LAYER_IT, 
			"Curr neighb.center=" << m_outIter.GetNeighborhood().m_currIndex);
#endif
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
			{ 
				//  Neighbors are same sign OR at least one neighbor is zero.
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
	m_updateBufferArray.SetArray(m_stepConfig->updateBuffBegin);
	m_layerIterator.SetBeginEnd(m_stepConfig->layer0Begin, m_stepConfig->layer0End);
	

	m_valueNeighbPreloader.Init();
	m_featureNeighbPreloader.Init();

	
	// prepare neighbour preloaders
	m_valueNeighbPreloader.SetImageProps(& m_Conf->valueImageProps);
	m_featureNeighbPreloader.SetImageProps(& m_Conf->featureImageProps);
	
	// Calculates the update values for the active layer indicies in this
	// iteration.  Iterates through the active layer index list, applying 
	// the level set function to the output image (level set image) at each
	// index.  Update values are stored in the update buffer.
	LayerNodeType *loaded;
	
	uint32 counter = 0;
	
	// first step in flow scenario - load the first
	loaded = m_layerIterator.GetLoaded();
		// load approp neigborhood
		m_valueNeighbPreloader.Load(*loaded);
		m_featureNeighbPreloader.Load(*loaded);
	 
	while(m_valueNeighbPreloader.GetCurrNodesNext() != m_stepConfig->layer0End)
	{
		loaded = m_layerIterator.GetLoaded();
			// load approp neigborhood
			m_valueNeighbPreloader.Load(*loaded);//->m_Value);
			m_featureNeighbPreloader.Load(*loaded);//->m_Value);

		
		m_outIter.SetNeighbourhood( m_valueNeighbPreloader.GetLoaded());
		m_featureIter.SetNeighbourhood( m_featureNeighbPreloader.GetLoaded());

#ifdef FOR_CELL
//		DL_PRINT(UPDATE_CALC_DEBUG, 
//						"node: " <<  << " value=" << m_outIter.GetCenterPixel());
#else
		DL_PRINT(UPDATE_CALC_DEBUG, 
				"node: " << loaded->m_Value << " value=" << m_outIter.GetCenterPixel());
#endif
				
		CalculateChangeItem();
		
		counter++;
	}
	
	m_updateBufferArray.FlushArray();
	
	// wait for ops to guarantee all is complete before this method ends
	// and to return its tags back to gate
	m_valueNeighbPreloader.Fini();
	m_featureNeighbPreloader.Fini();
	
#ifdef FOR_CELL	
	m_updateBufferArray.WaitForTransfer();
#endif

	// Ask the finite difference function to compute the time step for
	// this iteration.  We give it the global data pointer to use, then
	// ask it to free the global data memory.
	TimeStepType timeStep = m_diffFunc.ComputeGlobalTimeStep(&m_globalData);
	
	D_COMMAND(
			if(timeStep == 0) 
				D_PRINT("WARNING: CalculateChange: returning 0")
			);
	
#ifdef FOR_CELL
	DL_PRINT(UPDATE_CALC_DEBUG, "value=%f", timeStep);
#else
	DL_PRINT(UPDATE_CALC_DEBUG, "value=" << timeStep);
#endif
	
	return timeStep;
}

///////////////////////////////////////////////////////////////////////////////

