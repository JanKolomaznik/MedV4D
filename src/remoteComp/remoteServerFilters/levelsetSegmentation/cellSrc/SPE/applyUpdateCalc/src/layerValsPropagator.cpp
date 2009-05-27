
#include "common/Types.h"
#include "common/Log.h"

#include "../layerValsPropagator.h"

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////

LayerValuesPropagator::LayerValuesPropagator(SharedResources *shaRes)
	: m_propLayerValuesConfig(&shaRes->_propValConfig)
	, commonConf(&shaRes->_runConf)
{
	
}

///////////////////////////////////////////////////////////////////////////////

void LayerValuesPropagator::PropagateAllLayerValues()
{
	unsigned int i;

	// Update values in the first inside and first outside layers using the
	// active layer as a seed. Inside layers are odd numbers, outside layers are
	// even numbers. 
	_delta = -commonConf->m_ConstantGradientValue;
	this->PropagateLayerValues(0, 1, 3); // first inside
	_delta = commonConf->m_ConstantGradientValue;
	this->PropagateLayerValues(0, 2, 4); // first outside

	// Update the rest of the layers.
	for (i = 1; i < (uint32) LYERCOUNT - 2; ++i)
	{
		// Are we propagating values inward (more negative) or outward (more
		// positive)?
		if (((i+2)%2) == 1)
			_delta = -commonConf->m_ConstantGradientValue;
		else
			_delta = commonConf->m_ConstantGradientValue;
		this->PropagateLayerValues(i, i+2, i+4);
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerValuesPropagator::PropagateLayerValues(StatusType from, StatusType to,
		StatusType promote)
{	
	uint32 counter = 0;
	SparseFieldLevelSetNode *currNode;
	SparseFieldLevelSetNode *currNodeInLoadingNighbors;
	
	// prepare neighbour preloaders
	m_valueNeighPreloader.SetImageProps(&commonConf->valueImageProps);
	m_statusNeighPreloader.SetImageProps(&commonConf->statusImageProps);
	
	m_layerIterator.SetBeginEnd(
			m_propLayerValuesConfig->layerBegins[to], 
			m_propLayerValuesConfig->layerEnds[to]);
	
	if(m_layerIterator.HasNext())
	{
		currNode = m_layerIterator.Next();
		// load approp neigborhood
		m_valueNeighPreloader.Load(currNode->m_Value);
		m_statusNeighPreloader.Load(currNode->m_Value);
	}
	
	while (m_layerIterator.HasNext())
	{
		// load next portion
		if(m_layerIterator.HasNext())
		{
			currNodeInLoadingNighbors = m_layerIterator.Next();
			// load approp neigborhood
			m_valueNeighPreloader.Load(currNode->m_Value);
			m_statusNeighPreloader.Load(currNode->m_Value);
		}
		
		m_outIter.SetNeighbourhood( m_valueNeighPreloader.GetLoaded());
		m_statusIter.SetNeighbourhood( m_statusNeighPreloader.GetLoaded());
		
		counter++;

		DoTheWork(currNode, from, to, promote);
		
		// save to propagte changes in neighbourhoods
		m_valueNeighPreloader.SaveCurrItem();
		m_statusNeighPreloader.SaveCurrItem();
		
		currNode = currNodeInLoadingNighbors;
	}
	
	// NOTE: wait until save is complete ?
}

///////////////////////////////////////////////////////////////////////////////

void LayerValuesPropagator::DoTheWork(
		SparseFieldLevelSetNode *currNode, StatusType from, StatusType to,
		StatusType promote)
{
	unsigned int i;
	value = 0; // warnings
	bool found_neighbor_flag;
	StatusType past_end = static_cast<StatusType>( LYERCOUNT ) - 1;
	
	SparseFieldLevelSetNode *tmp;
	
	// Is this index marked for deletion? If the status image has
	// been marked with another layer's value, we need to delete this node
	// from the current list then skip to the next iteration.
	if (m_statusIter.GetCenterPixel() != to)
	{
		tmp = currNode;

		m_layerGate.UnlinkNode(
				this->m_layerIterator.GetCentralMemAddrrOfCurrProcessedNode(), to);
		//m_layerGate.ReturnToNodeStore(tmp);

		return;
	}

	found_neighbor_flag = false;
	for (i = 0; i < m_NeighborList.GetSize(); ++i)
	{
		// If this neighbor is in the "from" list, compare its absolute value
		// to to any previous values found in the "from" list.  Keep the value
		// that will cause the next layer to be closest to the zero level set.
		if (m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i) ) == from)
		{
			value_temp
					= m_outIter.GetPixel(m_NeighborList.GetArrayIndex(i) );

			if (found_neighbor_flag == false)
			{
				value = value_temp;
			}
			else
			{
				if (_delta < 0)
				{
					// Find the largest (least negative) neighbor
					if (value_temp > value)
					{
						value = value_temp;
					}
				}
				else
				{
					// Find the smallest (least positive) neighbor
					if (value_temp < value)
					{
						value = value_temp;
					}
				}
			}
			found_neighbor_flag = true;
		}
	}
	if (found_neighbor_flag == true)
	{
		// Set the new value using the smallest distance
		// found in our "from" neighbors.
		m_outIter.SetCenterPixel(value + _delta);
		//currNode = currNode->Next; // move on
	}
	else
	{
		// Did not find any neighbors on the "from" list, then promote this
		// node.  A "promote" value past the end of my sparse field size
		// means delete the node instead.  Change the status value in the
		// status image accordingly.
		tmp = currNode;
		//currNode = currNode->Next; // move on

		m_layerGate.UnlinkNode(
				this->m_layerIterator.GetCentralMemAddrrOfCurrProcessedNode(), to);
		if (promote > past_end)
		{
			//m_layerGate.ReturnToNodeStore(tmp);
			m_statusIter.SetCenterPixel(this->m_StatusNull);
		}
		else
		{
			m_layerGate.PushToLayer(tmp, promote);
			m_statusIter.SetCenterPixel(promote);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
