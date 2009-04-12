#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File pcPartOfTheFilter.tcc cannot be included directly!
#else

namespace itk {

///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PCPartOfSegmtLevelSetFilter()
{  
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::~PCPartOfSegmtLevelSetFilter()
{}


///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::SetupGate()
{
  for(uint32 i=0; i<this->m_Layers.size() ; i++)
  {
	  m_gateLayerPointers[i] = (M4D::Cell::LayerGate::LayerType *) this->m_Layers[i].GetPointer();
	  
	  applyUpdateCalc.conf.layerBegins[i] = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[i]->Begin().GetPointer();
	  applyUpdateCalc.conf.layerEnds[i] = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[i]->End().GetPointer();
  }
  applyUpdateCalc.SetGateProps(m_gateLayerPointers,
		  (M4D::Cell::LayerGate::LayerNodeStorageType *)this->m_LayerNodeStore.GetPointer() );
}
///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ApplyUpdate(TimeStepType dt)
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
  
	  LOUT << "ApplyUpdate" << std::endl << std::endl;
  
//  std::stringstream s;
//  s << "before" << this->m_ElapsedIterations << ".dat";
//  std::ofstream f(s.str().c_str());
//  PrintITKImage<OutputImageType>(*this->GetOutput(), f);
//  
//  LOUT << "Saving " << s << std::endl;
  
//  std::cout << "Update list (in ApplyUpdate):" << std::endl;
//  PrintUpdateBuf(std::cout);

  // Process the active layer.  This step will update the values in the active
  // layer as well as the values at indicies that *will* become part of the
  // active layer when they are promoted/demoted.  Also records promotions,
  // demotions in the m_StatusLayer for current active layer indicies
  // (i.e. those indicies which will move inside or outside the active
  // layers).
this->UpdateActiveLayerValues(dt, UpList[0], DownList[0]);
//  
//  std::stringstream s2;
//  s2 << "after" << this->m_ElapsedIterations << ".dat";
//  std::ofstream f2(s2.str().c_str());
//  PrintITKImage<OutputImageType>(*this->GetOutput(), f2);
  
//  LOUT << "Saving " << s2 << std::endl;

  // Process the status up/down lists.  This is an iterative process which
  // proceeds outwards from the active layer.  Each iteration generates the
  // list for the next iteration.
std::stringstream s;
	  s << "before" << this->m_ElapsedIterations;
//	  std::ofstream b(s.str().c_str());
//		PrintITKImage<StatusImageType>(*m_StatusImage.GetPointer(), b);
  
  // First process the status lists generated on the active layer.
  this->ProcessStatusList(UpList[0], UpList[1], 2, 1);
  this->ProcessStatusList(DownList[0], DownList[1], 1, 2);
  
  down_to = up_to = 0;
  up_search       = 3;
  down_search     = 4;
  j = 1;
  k = 0;
  while( down_search < static_cast<StatusType>( this->m_Layers.size() ) )
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
  this->ProcessStatusList(UpList[j], UpList[k], up_to, this->m_StatusNull);
  this->ProcessStatusList(DownList[j], DownList[k], down_to, this->m_StatusNull);
  
	  std::stringstream s2;
		  s2 << "beforeOutside" << this->m_ElapsedIterations;
//		  std::ofstream b1(s2.str().c_str());
//	  PrintITKImage<StatusImageType>(*m_StatusImage.GetPointer(), b1);
  // Now we are left with the lists of indicies which must be
  // brought into the outermost layers.  Bring UpList into last inside layer
  // and DownList into last outside layer.
  this->ProcessOutsideList(UpList[k], static_cast<int>( this->m_Layers.size()) -2);
  this->ProcessOutsideList(DownList[k], static_cast<int>( this->m_Layers.size()) -1);
	  

	std::stringstream s3;
		  s3 << "afterOutside" << this->m_ElapsedIterations;
//		  std::ofstream a1(s3.str().c_str());
//	PrintITKImage<StatusImageType>(*m_StatusImage.GetPointer(), a1);

  // Finally, we update all of the layer values (excluding the active layer,
  // which has already been updated).
  this->PropagateAllLayerValues();
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ProcessOutsideList(LayerType *OutsideList, StatusType ChangeToStatus)
{
  LayerNodeType *node;
  
  LOUT << "ProcessOutsideList" << std::endl << std::endl;
  
  // Push each index in the input list into its appropriate status layer
  // (ChangeToStatus) and update the status image value at that index.
  while ( ! OutsideList->Empty() )
    {
	  LOUT << "m_StatusImage->SetPixel(" << OutsideList->Front()->m_Value << ")=" << ((uint32)ChangeToStatus) << std::endl;
	  this->m_StatusImage->SetPixel(OutsideList->Front()->m_Value, ChangeToStatus); 
    node = OutsideList->Front();
    OutsideList->PopFront();
    this->m_Layers[ChangeToStatus]->PushFront(node);
    }
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ProcessStatusList(LayerType *InputList, LayerType *OutputList,
                    StatusType ChangeToStatus, StatusType SearchForStatus)
{
  unsigned int i;
  bool bounds_status;
  LayerNodeType *node;
  StatusType neighbor_status;
  NeighborhoodIterator<StatusImageType>
    statusIt(this->m_NeighborList.GetRadius(), this->m_StatusImage,
             this->GetOutput()->GetRequestedRegion());

  if (this->m_BoundsCheckingActive == false )
    {
    statusIt.NeedToUseBoundaryConditionOff();
    }
  
  LOUT << "ProcessStatusList" << std::endl << std::endl;
  
  // Push each index in the input list into its appropriate status layer
  // (ChangeToStatus) and update the status image value at that index.
  // Also examine the neighbors of the index to determine which need to go onto
  // the output list (search for SearchForStatus).
  while ( ! InputList->Empty() )
    {
	  node = InputList->Front();
    statusIt.SetLocation(node->m_Value);
    statusIt.SetCenterPixel(ChangeToStatus);

    LOUT << "1. node=" << node->m_Value << std::endl;
      
    InputList->PopFront();  // Must unlink from the input list  _before_ transferring to another list.
    this->m_Layers[ChangeToStatus]->PushFront(node);    
     
    for (i = 0; i < this->m_NeighborList.GetSize(); ++i)
      {
      neighbor_status = statusIt.GetPixel(this->m_NeighborList.GetArrayIndex(i));
      LOUT << "2. neighbor_status=" << ((uint32)neighbor_status) << std::endl;
      // Have we bumped up against the boundary?  If so, turn on bounds
      // checking.
      if ( neighbor_status == this->m_StatusBoundaryPixel )
        {
    	  this->m_BoundsCheckingActive = true;
        }

      if (neighbor_status == SearchForStatus)
        { // mark this pixel so we don't add it twice.
    	  LOUT << "3. neighbor_status == SearchForStatus" << std::endl;
        statusIt.SetPixel(this->m_NeighborList.GetArrayIndex(i),
        		this->m_StatusChanging, bounds_status);
        
        if (bounds_status == true)
          {
          node = this->m_LayerNodeStore->Borrow();
          node->m_Value = statusIt.GetIndex() +
            this->m_NeighborList.GetNeighborhoodOffset(i);
          
          LOUT << "4. pushing to outList node: " << node->m_Value << std::endl;
          OutputList->PushFront( node );
          } // else this index was out of bounds.
        }
      }
    }
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::UpdateActiveLayerValues(TimeStepType dt,
                          LayerType *UpList, LayerType *DownList)
{
	const ValueType LOWER_ACTIVE_THRESHOLD = - (this->m_ConstantGradientValue / 2.0);
	  const ValueType UPPER_ACTIVE_THRESHOLD =    this->m_ConstantGradientValue / 2.0;
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
	    outputIt(this->m_NeighborList.GetRadius(), this->GetOutput(),
	             this->GetOutput()->GetRequestedRegion());

	  NeighborhoodIterator<StatusImageType>
	    statusIt(this->m_NeighborList.GetRadius(), this->m_StatusImage,
	             this->GetOutput()->GetRequestedRegion());

	  if ( this->m_BoundsCheckingActive == false )
	    {
	    outputIt.NeedToUseBoundaryConditionOff();
	    statusIt.NeedToUseBoundaryConditionOff();
	    }
	  
//	  uint32 count = 0;
//  	  LOUT << "Active layer:" << std::endl;
//  	  for( layerIt = this->m_Layers[0]->Begin(); layerIt != this->m_Layers[0]->End(); layerIt=layerIt->Next, count++)
//  		  LOUT << layerIt->m_Value << ",";
//  	  LOUT << std::endl << "count=" << count << std::endl;
	  
	  counter =0;
	  rms_change_accumulator = this->m_ValueZero;
	  layerIt = this->m_Layers[0]->Begin();
	  updateIt = this->m_UpdateBuffer.begin();
	  while (layerIt != this->m_Layers[0]->End() )
	    {
	    outputIt.SetLocation(layerIt->m_Value);
	    statusIt.SetLocation(layerIt->m_Value);

	    new_value = this->CalculateUpdateValue(
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
	      for (i = 0; i < this->m_NeighborList.GetSize(); ++i)
	        {
	        if (statusIt.GetPixel(this->m_NeighborList.GetArrayIndex(i))
	            == this->m_StatusActiveChangingDown)
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
	      temp_value = new_value - this->m_ConstantGradientValue;
	      for (i = 0; i < this->m_NeighborList.GetSize(); ++i)
	        {
	        idx = this->m_NeighborList.GetArrayIndex(i);
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
	      node = this->m_LayerNodeStore->Borrow();
	      node->m_Value = layerIt->m_Value;
	      LOUT << "A1. pushing up node:" << node->m_Value << std::endl;
	      UpList->PushFront(node);
	      statusIt.SetCenterPixel(this->m_StatusActiveChangingUp);

	      // Now remove this index from the active list.
	      release_node = layerIt.GetPointer();
	      ++layerIt;
	      this->m_Layers[0]->Unlink(release_node);
	      this->m_LayerNodeStore->Return( release_node );
	      }

	    else if (new_value < LOWER_ACTIVE_THRESHOLD)
	      { // This index will move DOWN into a negative (inside) layer.

	      // First check for active layer neighbors moving in the opposite
	      // direction.
	      flag = false;
	      for (i = 0; i < this->m_NeighborList.GetSize(); ++i)
	        {
	        if (statusIt.GetPixel(this->m_NeighborList.GetArrayIndex(i))
	            == this->m_StatusActiveChangingUp)
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
	      temp_value = new_value + this->m_ConstantGradientValue;
	      for (i = 0; i < this->m_NeighborList.GetSize(); ++i)
	        {
	        idx = this->m_NeighborList.GetArrayIndex(i);
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
	      node = this->m_LayerNodeStore->Borrow();
	      node->m_Value = layerIt->m_Value;
	      LOUT << "A2. pushing down node:" << node->m_Value << std::endl;
	      DownList->PushFront(node);
	      statusIt.SetCenterPixel(this->m_StatusActiveChangingDown);

	      // Now remove this index from the active list.
	      release_node = layerIt.GetPointer();
	      ++layerIt;
	      this->m_Layers[0]->Unlink(release_node);
	      this->m_LayerNodeStore->Return( release_node );
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
	    { this->SetRMSChange(static_cast<double>(this->m_ValueZero)); }
	  else
	    {
	    this->SetRMSChange(static_cast<double>( vcl_sqrt((double)(rms_change_accumulator / static_cast<ValueType>(counter)) )) );
	    }
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::InitConfigStructures(void)
{	        
	Superclass::InitConfigStructures();
	
	#ifdef PC
	    SetupGate();
	    applyUpdateCalc.SetCommonConfiguration(&this->m_Conf);
	#endif
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
typename
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>::TimeStepType
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::CalculateChange()
{	  
	  TimeStepType dt;
	  
	  updateSolver.m_Conf = &this->m_Conf;
	  updateSolver.Init();
	  dt = updateSolver.CalculateChange();
	
	return dt;
}
///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PropagateAllLayerValues()
{
	  SetupGate();
	  
	  this->m_Conf.m_UpdateBufferData = &this->m_UpdateBuffer[0];
	  
	  applyUpdateCalc.PropagateAllLayerValues();
//  unsigned int i;
//
//  // Update values in the first inside and first outside layers using the
//  // active layer as a seed. Inside layers are odd numbers, outside layers are
//  // even numbers. 
//  this->PropagateLayerValues(0, 1, 3, 1); // first inside
//  this->PropagateLayerValues(0, 2, 4, 2); // first outside
//
//  // Update the rest of the layers.
//  for (i = 1; i < this->m_Layers.size() - 2; ++i)
//    {
//    this->PropagateLayerValues(i, i+2, i+4, (i+2)%2);
//    }
}
///////////////////////////////////////////////////////////////////////////////
//template<class TInputImage,class TFeatureImage, class TOutputPixelType>
//void
//PCPartOfSegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
//::PropagateLayerValues(StatusType from, StatusType to,
//                       StatusType promote, int InOrOut)
//{
//  unsigned int i;
//  ValueType value, value_temp, delta;
//  value = NumericTraits<ValueType>::Zero; // warnings
//  bool found_neighbor_flag;
//  typename LayerType::Iterator toIt;
//  LayerNodeType *node;
//  StatusType past_end = static_cast<StatusType>( this->m_Layers.size() ) - 1;
//  
//  // Are we propagating values inward (more negative) or outward (more
//  // positive)?
//  if (InOrOut == 1) delta = - this->m_ConstantGradientValue;
//  else delta = this->m_ConstantGradientValue;
// 
//  NeighborhoodIterator<OutputImageType>
//    outputIt(this->m_NeighborList.GetRadius(), this->GetOutput(),
//             this->GetOutput()->GetRequestedRegion() );
//  NeighborhoodIterator<StatusImageType>
//    statusIt(this->m_NeighborList.GetRadius(), this->m_StatusImage,
//             this->GetOutput()->GetRequestedRegion() );
//
//  if ( m_BoundsCheckingActive == false )
//    {
//    outputIt.NeedToUseBoundaryConditionOff();
//    statusIt.NeedToUseBoundaryConditionOff();
//    }
//  
//  toIt  = this->m_Layers[to]->Begin();
//  while ( toIt != this->m_Layers[to]->End() )
//    {
//    statusIt.SetLocation( toIt->m_Value );
//
//    // Is this index marked for deletion? If the status image has
//    // been marked with another layer's value, we need to delete this node
//    // from the current list then skip to the next iteration.
//    if (statusIt.GetCenterPixel() != to)
//      {
//      node = toIt.GetPointer();
//      ++toIt;
//      this->m_Layers[to]->Unlink( node );
//      this->m_LayerNodeStore->Return( node );
//      continue;
//      }
//      
//    outputIt.SetLocation( toIt->m_Value );
//
//    found_neighbor_flag = false;
//    for (i = 0; i < this->m_NeighborList.GetSize(); ++i)
//      {
//      // If this neighbor is in the "from" list, compare its absolute value
//      // to to any previous values found in the "from" list.  Keep the value
//      // that will cause the next layer to be closest to the zero level set.
//      if ( statusIt.GetPixel( this->m_NeighborList.GetArrayIndex(i) ) == from )
//        {
//        value_temp = outputIt.GetPixel( this->m_NeighborList.GetArrayIndex(i) );
//
//        if (found_neighbor_flag == false)
//          {
//          value = value_temp;
//          }
//        else
//          {
//          if (InOrOut == 1)
//            {
//            // Find the largest (least negative) neighbor
//            if ( value_temp > value )
//              {
//              value = value_temp;
//              }
//            }
//          else
//            {
//            // Find the smallest (least positive) neighbor
//            if (value_temp < value)
//              {
//              value = value_temp;
//              }
//            }
//          }
//        found_neighbor_flag = true;
//        }
//      }
//    if (found_neighbor_flag == true)
//      {
//      // Set the new value using the smallest distance
//      // found in our "from" neighbors.
//      outputIt.SetCenterPixel( value + delta );
//      ++toIt;
//      }
//    else
//      {
//      // Did not find any neighbors on the "from" list, then promote this
//      // node.  A "promote" value past the end of my sparse field size
//      // means delete the node instead.  Change the status value in the
//      // status image accordingly.
//      node  = toIt.GetPointer();
//      ++toIt;
//      this->m_Layers[to]->Unlink( node );
//      if ( promote > past_end )
//        {
//        this->m_LayerNodeStore->Return( node );
//        statusIt.SetCenterPixel(this->m_StatusNull);
//        }
//      else
//        {
//        this->m_Layers[promote]->PushFront( node );
//        statusIt.SetCenterPixel(promote);
//        }
//      }
//    }
//}

///////////////////////////////////////////////////////////////////////////////
}
#endif
