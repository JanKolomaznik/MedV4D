#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#error File SPEHardWorker.tcc cannot be included directly!
#else

namespace itk {


template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::ValueType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_ValueOne = NumericTraits<ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage,
                                                            TOutputImage>::ValueType >::One;

template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::ValueType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_ValueZero = NumericTraits<ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage,
                                                             TOutputImage>::ValueType >::Zero;

template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::StatusType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_StatusNull = NumericTraits<ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage,
                                                              TOutputImage>::StatusType >::NonpositiveMin();

template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::StatusType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_StatusChanging = -1;

template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::StatusType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_StatusActiveChangingUp = -2;

template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::StatusType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_StatusActiveChangingDown = -3;

template<class TInputImage, class TOutputImage>
ITK_TYPENAME SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>::StatusType
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::m_StatusBoundaryPixel = -4;

///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::ProcessStatusList(LayerType *InputList, LayerType *OutputList,
                    StatusType ChangeToStatus, StatusType SearchForStatus)
{
  unsigned int i;
  bool bounds_status;
  LayerNodeType *node;
  StatusType neighbor_status;
  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion());

  if (m_BoundsCheckingActive == false )
    {
    statusIt.NeedToUseBoundaryConditionOff();
    }
  
  // Push each index in the input list into its appropriate status layer
  // (ChangeToStatus) and update the status image value at that index.
  // Also examine the neighbors of the index to determine which need to go onto
  // the output list (search for SearchForStatus).
  while ( ! InputList->Empty() )
    {
    statusIt.SetLocation(InputList->Front()->m_Value);
    statusIt.SetCenterPixel(ChangeToStatus);

    node = InputList->Front();  // Must unlink from the input list 
    InputList->PopFront();      // _before_ transferring to another list.
    m_Layers[ChangeToStatus]->PushFront(node);
     
    for (i = 0; i < m_NeighborList.GetSize(); ++i)
      {
      neighbor_status = statusIt.GetPixel(m_NeighborList.GetArrayIndex(i));

      // Have we bumped up against the boundary?  If so, turn on bounds
      // checking.
      if ( neighbor_status == m_StatusBoundaryPixel )
        {
        m_BoundsCheckingActive = true;
        }

      if (neighbor_status == SearchForStatus)
        { // mark this pixel so we don't add it twice.
        statusIt.SetPixel(m_NeighborList.GetArrayIndex(i),
                          m_StatusChanging, bounds_status);
        if (bounds_status == true)
          {
          node = m_LayerNodeStore->Borrow();
          node->m_Value = statusIt.GetIndex() +
            m_NeighborList.GetNeighborhoodOffset(i);
          OutputList->PushFront( node );
          } // else this index was out of bounds.
        }
      }
    }
}
///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::UpdateActiveLayerValues(TimeStepType dt,
                          LayerType *UpList, LayerType *DownList)
{
  // This method scales the update buffer values by the time step and adds
  // them to the active layer pixels.  New values at an index which fall
  // outside of the active layer range trigger that index to be placed on the
  // "up" or "down" status list.  The neighbors of any such index are then
  // assigned new values if they are determined to be part of the active list
  // for the next iteration (i.e. their values will be raised or lowered into
  // the active range).
  const ValueType LOWER_ACTIVE_THRESHOLD = - (m_ConstantGradientValue / 2.0);
  const ValueType UPPER_ACTIVE_THRESHOLD =    m_ConstantGradientValue / 2.0;
  //   const ValueType LOWER_ACTIVE_THRESHOLD = - 0.7;
  //   const ValueType UPPER_ACTIVE_THRESHOLD =   0.7;
  ValueType new_value, temp_value, rms_change_accumulator;
  LayerNodeType *node, *release_node;
  StatusType neighbor_status;
  unsigned int i, idx, counter;
  bool bounds_status, flag;
  
  typename LayerType::Iterator         layerIt;
  typename UpdateBufferType::const_iterator updateIt;

  NeighborhoodIterator<OutputImageType>
    outputIt(m_NeighborList.GetRadius(), this->GetOutput(),
             this->GetOutput()->GetRequestedRegion());

  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion());

  if ( m_BoundsCheckingActive == false )
    {
    outputIt.NeedToUseBoundaryConditionOff();
    statusIt.NeedToUseBoundaryConditionOff();
    }
  
  counter =0;
  rms_change_accumulator = m_ValueZero;
  layerIt = m_Layers[0]->Begin();
  updateIt = m_UpdateBuffer.begin();
  while (layerIt != m_Layers[0]->End() )
    {
    outputIt.SetLocation(layerIt->m_Value);
    statusIt.SetLocation(layerIt->m_Value);

    new_value = this->CalculateUpdateValue(layerIt->m_Value,
                                           dt,
                                           outputIt.GetCenterPixel(),
                                           *updateIt);

    // If this index needs to be moved to another layer, then search its
    // neighborhood for indicies that need to be pulled up/down into the
    // active layer. Set those new active layer values appropriately,
    // checking first to make sure they have not been set by a more
    // influential neighbor.

    //   ...But first make sure any neighbors in the active layer are not
    // moving to a layer in the opposite direction.  This step is necessary
    // to avoid the creation of holes in the active layer.  The fix is simply
    // to not change this value and leave the index in the active set.

    if (new_value >= UPPER_ACTIVE_THRESHOLD)
      { // This index will move UP into a positive (outside) layer.

      // First check for active layer neighbors moving in the opposite
      // direction.
      flag = false;
      for (i = 0; i < m_NeighborList.GetSize(); ++i)
        {
        if (statusIt.GetPixel(m_NeighborList.GetArrayIndex(i))
            == m_StatusActiveChangingDown)
          {
          flag = true;
          break;
          }
        }
      if (flag == true)
        {
        ++layerIt;
        ++updateIt;
        continue;
        }

      rms_change_accumulator += vnl_math_sqr(new_value-outputIt.GetCenterPixel());

      // Search the neighborhood for inside indicies.
      temp_value = new_value - m_ConstantGradientValue;
      for (i = 0; i < m_NeighborList.GetSize(); ++i)
        {
        idx = m_NeighborList.GetArrayIndex(i);
        neighbor_status = statusIt.GetPixel( idx );
        if (neighbor_status == 1)
          {
          // Keep the smallest possible value for the new active node.  This
          // places the new active layer node closest to the zero level-set.
          if ( outputIt.GetPixel(idx) < LOWER_ACTIVE_THRESHOLD ||
               ::vnl_math_abs(temp_value) < ::vnl_math_abs(outputIt.GetPixel(idx)) )
            {
            outputIt.SetPixel(idx, temp_value, bounds_status);
            }
          }
        }
      node = m_LayerNodeStore->Borrow();
      node->m_Value = layerIt->m_Value;
      UpList->PushFront(node);
      statusIt.SetCenterPixel(m_StatusActiveChangingUp);

      // Now remove this index from the active list.
      release_node = layerIt.GetPointer();
      ++layerIt;
      m_Layers[0]->Unlink(release_node);
      m_LayerNodeStore->Return( release_node );
      }

    else if (new_value < LOWER_ACTIVE_THRESHOLD)
      { // This index will move DOWN into a negative (inside) layer.

      // First check for active layer neighbors moving in the opposite
      // direction.
      flag = false;
      for (i = 0; i < m_NeighborList.GetSize(); ++i)
        {
        if (statusIt.GetPixel(m_NeighborList.GetArrayIndex(i))
            == m_StatusActiveChangingUp)
          {
          flag = true;
          break;
          }
        }
      if (flag == true)
        {
        ++layerIt;
        ++updateIt;
        continue;
        }
      
      rms_change_accumulator += vnl_math_sqr(new_value - outputIt.GetCenterPixel());
          
      // Search the neighborhood for outside indicies.
      temp_value = new_value + m_ConstantGradientValue;
      for (i = 0; i < m_NeighborList.GetSize(); ++i)
        {
        idx = m_NeighborList.GetArrayIndex(i);
        neighbor_status = statusIt.GetPixel( idx );
        if (neighbor_status == 2)
          {
          // Keep the smallest magnitude value for this active set node.  This
          // places the node closest to the active layer.
          if ( outputIt.GetPixel(idx) >= UPPER_ACTIVE_THRESHOLD ||
               ::vnl_math_abs(temp_value) < ::vnl_math_abs(outputIt.GetPixel(idx)) )
            {
            outputIt.SetPixel(idx, temp_value, bounds_status);
            }
          }
        }
      node = m_LayerNodeStore->Borrow();
      node->m_Value = layerIt->m_Value;
      DownList->PushFront(node);
      statusIt.SetCenterPixel(m_StatusActiveChangingDown);

      // Now remove this index from the active list.
      release_node = layerIt.GetPointer();
      ++layerIt;
      m_Layers[0]->Unlink(release_node);
      m_LayerNodeStore->Return( release_node );
      }
    else
      {
      rms_change_accumulator += vnl_math_sqr(new_value - outputIt.GetCenterPixel());
      //rms_change_accumulator += (*updateIt) * (*updateIt);
      outputIt.SetCenterPixel( new_value );
      ++layerIt;
      }
    ++updateIt;
    ++counter;
    }
  
  // Determine the average change during this iteration.
  if (counter == 0)
    { this->SetRMSChange(static_cast<double>(m_ValueZero)); }
  else
    {
    this->SetRMSChange(static_cast<double>( vcl_sqrt((double)(rms_change_accumulator / static_cast<ValueType>(counter)) )) );
    }
}
///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::ProcessOutsideList(LayerType *OutsideList, StatusType ChangeToStatus)
{
  LayerNodeType *node;
  
  // Push each index in the input list into its appropriate status layer
  // (ChangeToStatus) and update the status image value at that index.
  while ( ! OutsideList->Empty() )
    {
    m_StatusImage->SetPixel(OutsideList->Front()->m_Value, ChangeToStatus); 
    node = OutsideList->Front();
    OutsideList->PopFront();
    m_Layers[ChangeToStatus]->PushFront(node);
    }
}
///////////////////////////////////////////////////////////////////////////////

void ProcessStep(linkedList begin, LinkedList end)
{
	unsigned int i, j, k, t;

	  StatusType up_to, up_search;
	  StatusType down_to, down_search;
	  
	  LayerPointerType UpList[2];
	  LayerPointerType DownList[2];
	  for (i = 0; i < 2; ++i)
	    {
	    UpList[i]   = LayerType::New();
	    DownList[i] = LayerType::New();
	    }
	  
	  // Process the active layer.  This step will update the values in the active
	  // layer as well as the values at indicies that *will* become part of the
	  // active layer when they are promoted/demoted.  Also records promotions,
	  // demotions in the m_StatusLayer for current active layer indicies
	  // (i.e. those indicies which will move inside or outside the active
	  // layers).
	  this->UpdateActiveLayerValues(dt, UpList[0], DownList[0]);

	  // Process the status up/down lists.  This is an iterative process which
	  // proceeds outwards from the active layer.  Each iteration generates the
	  // list for the next iteration.
	  
	  // First process the status lists generated on the active layer.
	  this->ProcessStatusList(UpList[0], UpList[1], 2, 1);
	  this->ProcessStatusList(DownList[0], DownList[1], 1, 2);
	  
	  down_to = up_to = 0;
	  up_search       = 3;
	  down_search     = 4;
	  j = 1;
	  k = 0;
	  while( down_search < static_cast<StatusType>( m_Layers.size() ) )
	    {
	    this->ProcessStatusList(UpList[j], UpList[k], up_to, up_search);
	    this->ProcessStatusList(DownList[j], DownList[k], down_to, down_search);

	    if (up_to == 0) up_to += 1;
	    else            up_to += 2;
	    down_to += 2;

	    up_search += 2;
	    down_search += 2;

	    // Swap the lists so we can re-use the empty one.
	    t = j;
	    j = k;
	    k = t;
	    }

	  // Process the outermost inside/outside layers in the sparse field.
	  this->ProcessStatusList(UpList[j], UpList[k], up_to, m_StatusNull);
	  this->ProcessStatusList(DownList[j], DownList[k], down_to, m_StatusNull);
	  
	  // Now we are left with the lists of indicies which must be
	  // brought into the outermost layers.  Bring UpList into last inside layer
	  // and DownList into last outside layer.
	  this->ProcessOutsideList(UpList[k], static_cast<int>( m_Layers.size()) -2);
	  this->ProcessOutsideList(DownList[k], static_cast<int>( m_Layers.size()) -1);

	  // Finally, we update all of the layer values (excluding the active layer,
	  // which has already been updated).
	  this->PropagateAllLayerValues();
}

///////////////////////////////////////////////////////////////////////////////

CalculateChange()
{
  const typename Superclass::FiniteDifferenceFunctionType::Pointer df
    = this->GetDifferenceFunction();
  typename Superclass::FiniteDifferenceFunctionType::FloatOffsetType offset;
  ValueType norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
    backwardValue, centerValue;
  unsigned i;
  ValueType MIN_NORM      = 1.0e-6;
  if (this->GetUseImageSpacing())
    {
    double minSpacing = NumericTraits<double>::max();
    for (i=0; i<ImageDimension; i++)
      {
      minSpacing = vnl_math_min(minSpacing,this->GetInput()->GetSpacing()[i]);
      }
    MIN_NORM *= minSpacing;
    }

  void *globalData = df->GetGlobalDataPointer();
  
  typename LayerType::ConstIterator layerIt;
  NeighborhoodIterator<OutputImageType> outputIt(df->GetRadius(),
                    this->GetOutput(), this->GetOutput()->GetRequestedRegion());
  TimeStepType timeStep;

  const NeighborhoodScalesType neighborhoodScales = this->GetDifferenceFunction()->ComputeNeighborhoodScales();

  if ( m_BoundsCheckingActive == false )
    {
    outputIt.NeedToUseBoundaryConditionOff();
    }
  
  m_UpdateBuffer.clear();
  m_UpdateBuffer.reserve(m_Layers[0]->Size());

  // Calculates the update values for the active layer indicies in this
  // iteration.  Iterates through the active layer index list, applying 
  // the level set function to the output image (level set image) at each
  // index.  Update values are stored in the update buffer.
  for (layerIt = m_Layers[0]->Begin(); layerIt != m_Layers[0]->End(); ++layerIt)
    {
    outputIt.SetLocation(layerIt->m_Value);

    // Calculate the offset to the surface from the center of this
    // neighborhood.  This is used by some level set functions in sampling a
    // speed, advection, or curvature term.
    if(centerValue = outputIt.GetCenterPixel()) != 0.0 )
      {
      // Surface is at the zero crossing, so distance to surface is:
      // phi(x) / norm(grad(phi)), where phi(x) is the center of the
      // neighborhood.  The location is therefore
      // (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
      norm_grad_phi_squared = 0.0;
      for (i = 0; i < ImageDimension; ++i)
        {
        forwardValue  = outputIt.GetNext(i);
        backwardValue = outputIt.GetPrevious(i);
            
        if (forwardValue * backwardValue >= 0)
          { //  Neighbors are same sign OR at least one neighbor is zero.
          dx_forward  = forwardValue - centerValue;
          dx_backward = centerValue - backwardValue;

          // Pick the larger magnitude derivative.
          if (::vnl_math_abs(dx_forward) > ::vnl_math_abs(dx_backward) )
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
      
      for (i = 0; i < ImageDimension; ++i)
        {
        offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
        }
          
      m_UpdateBuffer.push_back( df->ComputeUpdate(outputIt, globalData, offset) );
      }
    }
  
  // Ask the finite difference function to compute the time step for
  // this iteration.  We give it the global data pointer to use, then
  // ask it to free the global data memory.
  timeStep = df->ComputeGlobalTimeStep(globalData);

  df->ReleaseGlobalDataPointer(globalData);
  
  return timeStep;
}

///////////////////////////////////////////////////////////////////////////////

}  // namespace itk

#endif