
#include "common/Types.h"
#include "../applyUpdateCalculator.h"
#include "../../vnl_math.h"
#include <string.h>

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////
void
ApplyUpdateSPE::PropagateAllLayerValues()
{
  unsigned int i;

  // Update values in the first inside and first outside layers using the
  // active layer as a seed. Inside layers are odd numbers, outside layers are
  // even numbers. 
  this->PropagateLayerValues(0, 1, 3, 1); // first inside
  this->PropagateLayerValues(0, 2, 4, 2); // first outside

  // Update the rest of the layers.
  for (i = 1; i < commonConf.m_NumberOfLayers - 2; ++i)
    {
    this->PropagateLayerValues(i, i+2, i+4, (i+2)%2);
    }
}
///////////////////////////////////////////////////////////////////////////////
void
ApplyUpdateSPE::PropagateLayerValues(StatusType from, StatusType to,
                       StatusType promote, int InOrOut)
{
  unsigned int i;
  ValueType value, value_temp, delta;
  value = NumericTraits<ValueType>::Zero; // warnings
  bool found_neighbor_flag;
  typename LayerType::Iterator toIt;
  LayerNodeType *node;
  StatusType past_end = static_cast<StatusType>( m_Layers.size() ) - 1;
  
  // Are we propagating values inward (more negative) or outward (more
  // positive)?
  if (InOrOut == 1) delta = - m_ConstantGradientValue;
  else delta = m_ConstantGradientValue;
 
  NeighborhoodIterator<OutputImageType>
    outputIt(m_NeighborList.GetRadius(), this->GetOutput(),
             this->GetOutput()->GetRequestedRegion() );
  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion() );

  if ( m_BoundsCheckingActive == false )
    {
    outputIt.NeedToUseBoundaryConditionOff();
    statusIt.NeedToUseBoundaryConditionOff();
    }
  
  toIt  = m_Layers[to]->Begin();
  while ( toIt != m_Layers[to]->End() )
    {
    statusIt.SetLocation( toIt->m_Value );

    // Is this index marked for deletion? If the status image has
    // been marked with another layer's value, we need to delete this node
    // from the current list then skip to the next iteration.
    if (statusIt.GetCenterPixel() != to)
      {
      node = toIt.GetPointer();
      ++toIt;
      m_Layers[to]->Unlink( node );
      m_LayerNodeStore->Return( node );
      continue;
      }
      
    outputIt.SetLocation( toIt->m_Value );

    found_neighbor_flag = false;
    for (i = 0; i < m_NeighborList.GetSize(); ++i)
      {
      // If this neighbor is in the "from" list, compare its absolute value
      // to to any previous values found in the "from" list.  Keep the value
      // that will cause the next layer to be closest to the zero level set.
      if ( statusIt.GetPixel( m_NeighborList.GetArrayIndex(i) ) == from )
        {
        value_temp = outputIt.GetPixel( m_NeighborList.GetArrayIndex(i) );

        if (found_neighbor_flag == false)
          {
          value = value_temp;
          }
        else
          {
          if (InOrOut == 1)
            {
            // Find the largest (least negative) neighbor
            if ( value_temp > value )
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
      outputIt.SetCenterPixel( value + delta );
      ++toIt;
      }
    else
      {
      // Did not find any neighbors on the "from" list, then promote this
      // node.  A "promote" value past the end of my sparse field size
      // means delete the node instead.  Change the status value in the
      // status image accordingly.
      node  = toIt.GetPointer();
      ++toIt;
      m_Layers[to]->Unlink( node );
      if ( promote > past_end )
        {
        m_LayerNodeStore->Return( node );
        statusIt.SetCenterPixel(this->m_StatusNull);
        }
      else
        {
        m_Layers[promote]->PushFront( node );
        statusIt.SetCenterPixel(promote);
        }
      }
    }
}

///////////////////////////////////////////////////////////////////////////////

void
ApplyUpdateSPE::UnlinkNode(SparseFieldLevelSetNode *node)
{
	
}

///////////////////////////////////////////////////////////////////////////////