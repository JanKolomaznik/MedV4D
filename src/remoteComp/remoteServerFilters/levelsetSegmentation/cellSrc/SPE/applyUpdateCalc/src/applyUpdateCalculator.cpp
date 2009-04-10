#include "common/Types.h"
#include "common/Common.h"
#include "../applyUpdateCalculator.h"
#include "../../vnl_math.h"
#include <math.h>	//sqrt
#include <string.h>

using namespace M4D::Cell;

#define LYERCOUNT(X) ((X * 2) + 1)

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ApplyUpdateSPE()
{

}

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::~ApplyUpdateSPE()
{

}

///////////////////////////////////////////////////////////////////////////////
void ApplyUpdateSPE::PropagateAllLayerValues()
{
	unsigned int i;

	// Update values in the first inside and first outside layers using the
	// active layer as a seed. Inside layers are odd numbers, outside layers are
	// even numbers. 
	this->PropagateLayerValues(0, 1, 3, 1); // first inside
	this->PropagateLayerValues(0, 2, 4, 2); // first outside

	// Update the rest of the layers.
	for (i = 1; i < (uint32)LYERCOUNT(commonConf->m_NumberOfLayers) - 2; ++i)
	{
		this->PropagateLayerValues(i, i+2, i+4, (i+2)%2);
	}
}
///////////////////////////////////////////////////////////////////////////////
void ApplyUpdateSPE::PropagateLayerValues(StatusType from, StatusType to,
		StatusType promote, uint32 InOrOut)
{
	unsigned int i;
	TPixelValue value, value_temp, delta;
	value = 0; // warnings
	bool found_neighbor_flag;
	StatusType past_end = static_cast<StatusType>( LYERCOUNT(commonConf->m_NumberOfLayers) ) - 1;

	SparseFieldLevelSetNode *tmp;

	// Are we propagating values inward (more negative) or outward (more
	// positive)?
	if (InOrOut == 1)
		delta = -commonConf->m_ConstantGradientValue;
	else
		delta = commonConf->m_ConstantGradientValue;

	//  NeighborhoodIterator<OutputImageType>
	//    outputIt(m_NeighborList.GetRadius(), this->GetOutput(),
	//             this->GetOutput()->GetRequestedRegion() );
	//  NeighborhoodIterator<StatusImageType>
	//    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
	//             this->GetOutput()->GetRequestedRegion() );

	NeighborhoodCell<TPixelValue> outNeigh( &commonConf->valueImageProps);
	NeighborhoodCell<StatusType> statusNeigh( &commonConf->statusImageProps);

	m_outIter.SetNeighbourhood( &outNeigh);
	m_statusIter.SetNeighbourhood( &statusNeigh);

	SparseFieldLevelSetNode *currNode = conf.layerBegins[to];
	while (currNode != conf.layerEnds[to])
	{
		m_statusIter.SetLocation(currNode->m_Value);

		// Is this index marked for deletion? If the status image has
		// been marked with another layer's value, we need to delete this node
		// from the current list then skip to the next iteration.
		if (m_statusIter.GetCenterPixel() != to)
		{
			tmp = currNode;
			currNode = currNode->Next; // move on

			UnlinkNode(tmp, to);
			ReturnToNodeStore(tmp);

			continue;
		}

		m_outIter.SetLocation(currNode->m_Value);

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
					if (InOrOut == 1)
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
			m_outIter.SetCenterPixel(value + delta);
			currNode = currNode->Next; // move on
		}
		else
		{
			// Did not find any neighbors on the "from" list, then promote this
			// node.  A "promote" value past the end of my sparse field size
			// means delete the node instead.  Change the status value in the
			// status image accordingly.
			tmp = currNode;
			currNode = currNode->Next; // move on

			UnlinkNode(tmp, to);
			if (promote > past_end)
			{
				ReturnToNodeStore(tmp);
				m_statusIter.SetCenterPixel(this->m_StatusNull);
			}
			else
			{
				PushToLayer(tmp, promote);
				m_statusIter.SetCenterPixel(promote);
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

void ApplyUpdateSPE::UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	m_Layers[layerNum]->Unlink(node);
}

///////////////////////////////////////////////////////////////////////////////

void ApplyUpdateSPE::ReturnToNodeStore(SparseFieldLevelSetNode *node)
{
	m_LayerNodeStore->Return(node);
}

///////////////////////////////////////////////////////////////////////////////

void ApplyUpdateSPE::PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	m_Layers[layerNum]->PushFront(node);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ValueType
ApplyUpdateSPE::ApplyUpdate(TimeStepType dt)
{
	unsigned int i, j, k, t;

	StatusType up_to, up_search;
	StatusType down_to, down_search;

	LayerPointerType UpList[2];
	LayerPointerType DownList[2];
	for (i = 0; i < 2; ++i)
	{
		UpList[i] = LayerType::New();
		DownList[i] = LayerType::New();
	}

	//  LOG("Update list:");
	//  for(typename UpdateBufferType::iterator it = m_UpdateBuffer.begin(); it != m_UpdateBuffer.end(); it++)
	//	  LOUT << *it << ", ";
	//  LOG("");
	
	NeighborhoodCell<TPixelValue> outNeigh( &commonConf->valueImageProps);
	NeighborhoodCell<StatusType> statusNeigh( &commonConf->statusImageProps);

	m_outIter.SetNeighbourhood( &outNeigh);
	m_statusIter.SetNeighbourhood( &statusNeigh);

	// Process the active layer.  This step will update the values in the active
	// layer as well as the values at indicies that *will* become part of the
	// active layer when they are promoted/demoted.  Also records promotions,
	// demotions in the m_StatusLayer for current active layer indicies
	// (i.e. those indicies which will move inside or outside the active
	// layers).
	ValueType retval = this->UpdateActiveLayerValues(dt, UpList[0], DownList[0]);//, m_outIter, m_statusIter);

	// Process the status up/down lists.  This is an iterative process which
	// proceeds outwards from the active layer.  Each iteration generates the
	// list for the next iteration.

	// First process the status lists generated on the active layer.
	this->ProcessStatusList(UpList[0], UpList[1], 2, 1, m_statusIter);
	this->ProcessStatusList(DownList[0], DownList[1], 1, 2, m_statusIter);

	down_to = up_to = 0;
	up_search = 3;
	down_search = 4;
	j = 1;
	k = 0;
	while (down_search < static_cast<StatusType>( LYERCOUNT(commonConf->m_NumberOfLayers) ) )
	{
		this->ProcessStatusList(UpList[j], UpList[k], up_to, up_search, m_statusIter);
		this->ProcessStatusList(DownList[j], DownList[k], down_to, down_search, m_statusIter);

		if (up_to == 0)
			up_to += 1;
		else
			up_to += 2;
		down_to += 2;

		up_search += 2;
		down_search += 2;

		// Swap the lists so we can re-use the empty one.
		t = j;
		j = k;
		k = t;
	}

	// Process the outermost inside/outside layers in the sparse field.
	this->ProcessStatusList(UpList[j], UpList[k], up_to, this->m_StatusNull, m_statusIter);
	this->ProcessStatusList(DownList[j], DownList[k], down_to,
			this->m_StatusNull, m_statusIter);

	// Now we are left with the lists of indicies which must be
	// brought into the outermost layers.  Bring UpList into last inside layer
	// and DownList into last outside layer.
	this->ProcessOutsideList(UpList[k], static_cast<int>(LYERCOUNT(commonConf->m_NumberOfLayers)) -2, m_statusIter);
	this->ProcessOutsideList(DownList[k], static_cast<int>(LYERCOUNT(commonConf->m_NumberOfLayers)) -1, m_statusIter);

	// Finally, we update all of the layer values (excluding the active layer,
	// which has already been updated).
	this->PropagateAllLayerValues();
	
	return retval;
}

///////////////////////////////////////////////////////////////////////////////

void
ApplyUpdateSPE::ProcessOutsideList(
		LayerType *OutsideList, StatusType ChangeToStatus, TStatusNeighbIterator &statIter)
{
	SparseFieldLevelSetNode *node;

	// Push each index in the input list into its appropriate status layer
	// (ChangeToStatus) and update the status image value at that index.
	while ( !OutsideList->Empty() )
	{
		statIter.SetLocation(OutsideList->Front()->m_Value);
		statIter.SetCenterPixel(ChangeToStatus);
		node = OutsideList->Front();
		OutsideList->PopFront();
		m_Layers[ChangeToStatus]->PushFront(node);
	}
}
///////////////////////////////////////////////////////////////////////////////
void
ApplyUpdateSPE::ProcessStatusList(
		LayerType *InputList, LayerType *OutputList, StatusType ChangeToStatus,
		StatusType SearchForStatus, TStatusNeighbIterator &statusIt)
{
	unsigned int i;
	bool bounds_status;
	SparseFieldLevelSetNode *node;
	StatusType neighbor_status;
//	NeighborhoodIterator<StatusImageType> statusIt(m_NeighborList.GetRadius(),
//			m_StatusImage, this->GetOutput()->GetRequestedRegion());
		

	// Push each index in the input list into its appropriate status layer
	// (ChangeToStatus) and update the status image value at that index.
	// Also examine the neighbors of the index to determine which need to go onto
	// the output list (search for SearchForStatus).
	while ( !InputList->Empty() )
	{
		statusIt.SetLocation(InputList->Front()->m_Value);
		statusIt.SetCenterPixel(ChangeToStatus);

		node = InputList->Front(); // Must unlink from the input list 
		InputList->PopFront(); // _before_ transferring to another list.
		m_Layers[ChangeToStatus]->PushFront(node);

		for (i = 0; i < m_NeighborList.GetSize(); ++i)
		{
			neighbor_status
					= statusIt.GetPixel(m_NeighborList.GetArrayIndex(i));

			if (neighbor_status == SearchForStatus)
			{ // mark this pixel so we don't add it twice. //TODO
				statusIt.SetPixel(m_NeighborList.GetNeighborhoodOffset(i),
						this->m_StatusChanging);
				if (bounds_status == true)
				{
					node = m_LayerNodeStore->Borrow();
					node->m_Value = statusIt.GetIndex()
							+ m_NeighborList.GetNeighborhoodOffset(i);
					OutputList->PushFront(node);
				} // else this index was out of bounds.
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ValueType 
ApplyUpdateSPE::UpdateActiveLayerValues(
		TimeStepType dt, LayerType *UpList, LayerType *DownList)
{
//	const ValueType LOWER_ACTIVE_THRESHOLD = - (commonConf->m_ConstantGradientValue / 2.0);
//	const ValueType UPPER_ACTIVE_THRESHOLD = commonConf->m_ConstantGradientValue / 2.0;
//	//   const ValueType LOWER_ACTIVE_THRESHOLD = - 0.7;
//	//   const ValueType UPPER_ACTIVE_THRESHOLD =   0.7;
//	TPixelValue new_value, temp_value, rms_change_accumulator;
//	SparseFieldLevelSetNode *node, *release_node;
//	StatusType neighbor_status;
//	unsigned int i, idx, counter;
//	bool flag;//bounds_status, 
//
//	LayerType::Iterator layerIt;
//
////	NeighborhoodIterator<OutputImageType> outputIt(m_NeighborList.GetRadius(),
////			this->GetOutput(), this->GetOutput()->GetRequestedRegion());
////
////	NeighborhoodIterator<StatusImageType> statusIt(m_NeighborList.GetRadius(),
////			m_StatusImage, this->GetOutput()->GetRequestedRegion());
//	
//	NeighborhoodCell<TPixelValue> outNeigh( &commonConf->valueImageProps);
//	NeighborhoodCell<StatusType> statusNeigh( &commonConf->statusImageProps);
//
//	m_outIter.SetNeighbourhood( &outNeigh);
//	m_statusIter.SetNeighbourhood( &statusNeigh);
//
//	counter =0;
//	rms_change_accumulator = this->m_ValueZero;
//	layerIt = m_Layers[0]->Begin();
//	
//	TUpdateBufferArray updateBufferArray(commonConf->m_UpdateBufferData);
//	
//	std::cout << "updated activeLayerVals:" << std::endl;
//	
//	while (layerIt != m_Layers[0]->End() )
//	{
//		m_outIter.SetLocation(layerIt->m_Value);
//		m_statusIter.SetLocation(layerIt->m_Value);
//
//		new_value = this->CalculateUpdateValue(
//				dt, m_outIter.GetCenterPixel(), updateBufferArray.pop_front() );
//		
//		std::cout << new_value << ", ";
//
//		// If this index needs to be moved to another layer, then search its
//		// neighborhood for indicies that need to be pulled up/down into the
//		// active layer. Set those new active layer values appropriately,
//		// checking first to make sure they have not been set by a more
//		// influential neighbor.
//
//		//   ...But first make sure any neighbors in the active layer are not
//		// moving to a layer in the opposite direction.  This step is necessary
//		// to avoid the creation of holes in the active layer.  The fix is simply
//		// to not change this value and leave the index in the active set.
//
//		if (new_value >= UPPER_ACTIVE_THRESHOLD)
//		{ // This index will move UP into a positive (outside) layer.
//
//			// First check for active layer neighbors moving in the opposite
//			// direction.
//			flag = false;
//			for (i = 0; i < m_NeighborList.GetSize(); ++i)
//			{
//				if (m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i))
//						== this->m_StatusActiveChangingDown)
//				{
//					flag = true;
//					break;
//				}
//			}
//			if (flag == true)
//			{
//				++layerIt;
//				continue;
//			}
//
//			rms_change_accumulator += vnl_math_sqr(new_value
//					-m_outIter.GetCenterPixel());
//
//			// Search the neighborhood for inside indicies.
//			temp_value = new_value - commonConf->m_ConstantGradientValue;
//			for (i = 0; i < m_NeighborList.GetSize(); ++i)
//			{
//				idx = m_NeighborList.GetArrayIndex(i);
//				neighbor_status = m_statusIter.GetPixel(idx);
//				if (neighbor_status == 1)
//				{
//					// Keep the smallest possible value for the new active node.  This
//					// places the new active layer node closest to the zero level-set.
//					if (m_outIter.GetPixel(idx) < LOWER_ACTIVE_THRESHOLD ||:: vnl_math_abs(temp_value) < ::vnl_math_abs(m_outIter.GetPixel(idx)) )
//            {
//            m_outIter.SetPixel(m_NeighborList.GetNeighborhoodOffset(idx), temp_value);
//            }
//          }
//        }
//      node = m_LayerNodeStore->Borrow();
//      node->m_Value = layerIt->m_Value;
//      UpList->PushFront(node);
//      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingUp);
//
//      // Now remove this index from the active list.
//      release_node = layerIt.GetPointer();
//      ++layerIt;
//      m_Layers[0]->Unlink(release_node);
//      m_LayerNodeStore->Return( release_node );
//      }
//
//    else if (new_value < LOWER_ACTIVE_THRESHOLD)
//      { // This index will move DOWN into a negative (inside) layer.
//
//      // First check for active layer neighbors moving in the opposite
//      // direction.
//      flag = false;
//      for (i = 0; i < m_NeighborList.GetSize(); ++i)
//        {
//        if (m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i))
//            == this->m_StatusActiveChangingUp)
//          {
//          flag = true;
//          break;
//          }
//        }
//      if (flag == true)
//        {
//        ++layerIt;
//        continue;
//        }
//      
//      rms_change_accumulator += vnl_math_sqr(new_value - m_outIter.GetCenterPixel());
//          
//      // Search the neighborhood for outside indicies.
//      temp_value = new_value + commonConf->m_ConstantGradientValue;
//      for (i = 0; i < m_NeighborList.GetSize(); ++i)
//        {
//        idx = m_NeighborList.GetArrayIndex(i);
//        neighbor_status = m_statusIter.GetPixel( idx );
//        if (neighbor_status == 2)
//          {
//          // Keep the smallest magnitude value for this active set node.  This
//          // places the node closest to the active layer.
//          if ( m_outIter.GetPixel(idx) >= UPPER_ACTIVE_THRESHOLD ||
//               ::vnl_math_abs(temp_value) < ::vnl_math_abs(m_outIter.GetPixel(idx)) )
//            {
//            m_outIter.SetPixel(m_NeighborList.GetNeighborhoodOffset(idx), temp_value);
//            }
//          }
//        }
//      node = m_LayerNodeStore->Borrow();
//      node->m_Value = layerIt->m_Value;
//      DownList->PushFront(node);
//      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingDown);
//
//      // Now remove this index from the active list.
//      release_node = layerIt.GetPointer();
//      ++layerIt;
//      m_Layers[0]->Unlink(release_node);
//      m_LayerNodeStore->Return( release_node );
//      }
//    else
//      {
//      rms_change_accumulator += vnl_math_sqr(new_value - m_outIter.GetCenterPixel());
//      //rms_change_accumulator += (*updateIt) * (*updateIt);
//      m_outIter.SetCenterPixel( new_value );
//      ++layerIt;
//      }
//    ++counter;
//    }
	
	  const ValueType LOWER_ACTIVE_THRESHOLD = - (commonConf->m_ConstantGradientValue / 2.0);
	  const ValueType UPPER_ACTIVE_THRESHOLD =    commonConf->m_ConstantGradientValue / 2.0;
	  //   const ValueType LOWER_ACTIVE_THRESHOLD = - 0.7;
	  //   const ValueType UPPER_ACTIVE_THRESHOLD =   0.7;
	  ValueType new_value, temp_value, rms_change_accumulator;
	  SparseFieldLevelSetNode *node, *release_node;
	  StatusType neighbor_status;
	  unsigned int i, idx, counter;
	  bool bounds_status, flag;
	  
	  LayerType::Iterator         layerIt;
	  
	  TPixelValue *updateIt = commonConf->m_UpdateBufferData;
	  
	  NeighborhoodCell<TPixelValue> outNeigh( &commonConf->valueImageProps);
  	NeighborhoodCell<StatusType> statusNeigh( &commonConf->statusImageProps);
  	
  	m_outIter.SetNeighbourhood( &outNeigh);
  		m_statusIter.SetNeighbourhood( &statusNeigh);
	  
  		//LOUT << "updated activeLayerVals:" << std::endl;
  		
//  	    ValueType centerVal = m_outIter.GetCenterPixel();
//  	LOUT << "centerVal = " << centerVal << std::endl;
	  
	  TIndex currIndex;
	  
	  counter =0;
	  rms_change_accumulator = this->m_ValueZero;
	  layerIt = m_Layers[0]->Begin();
	  while (layerIt != m_Layers[0]->End() )
	    {
		  m_outIter.SetLocation(layerIt->m_Value);
		  m_statusIter.SetLocation(layerIt->m_Value);
		  
		  currIndex = layerIt->m_Value;
		  
		  TPixelValue curVal = m_outIter.GetCenterPixel();
	
	    new_value = this->CalculateUpdateValue(dt, curVal, *updateIt);
	    
	    //LOUT << new_value << ", ";
	    
	    ValueType centerVal = m_outIter.GetCenterPixel();
	LOUT << "1centerVal = " << centerVal << std::endl;
	
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
LOUT << "2new_value = " << new_value << ">= UPPER_ACTIVE_THRESHOLD" << std::endl;
	      // First check for active layer neighbors moving in the opposite
	      // direction.
	      flag = false;
	      for (i = 0; i < m_NeighborList.GetSize(); ++i)
	        {
	        if (m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i))
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
	        LOUT << "3flag == true" << std::endl;
	        continue;
	        }
	
	      rms_change_accumulator += vnl_math_sqr(new_value- centerVal );
	      LOUT << "4rms_change_accumulator = " << rms_change_accumulator << std::endl;
	
	      // Search the neighborhood for inside indicies.
	      temp_value = new_value - commonConf->m_ConstantGradientValue;
	      LOUT << "5temp_value = " << temp_value << std::endl;
	      for (i = 0; i < m_NeighborList.GetSize(); ++i)
	        {
	        idx = m_NeighborList.GetArrayIndex(i);
	        neighbor_status = m_statusIter.GetPixel( idx );
	        char tmp[2]; tmp[1] = 0;
	               tmp[0] = (neighbor_status + '0');
	        LOUT << "333status on " << idx << ":" << tmp << std::endl;
	        if (neighbor_status == 1)
	          {
	          // Keep the smallest possible value for the new active node.  This
	          // places the new active layer node closest to the zero level-set.
	        	ValueType rr = m_outIter.GetPixel(idx);
	        	LOUT << "444neighbor_status == 1, pix=" << rr << std::endl;
	          if ( rr < LOWER_ACTIVE_THRESHOLD ||
	               ::vnl_math_abs(temp_value) < ::vnl_math_abs(rr) )
	            {
	            m_outIter.SetPixel(idx, temp_value);
	            }
	          }
	        }
	      node = m_LayerNodeStore->Borrow();
	      LOUT << "6m_LayerNodeStore->Borrow() = " << std::endl;
	      node->m_Value = layerIt->m_Value;
	      UpList->PushFront(node);
	      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingUp);
	
	      // Now remove this index from the active list.
	      release_node = layerIt.GetPointer();
	      ++layerIt;
	      m_Layers[0]->Unlink(release_node);
	      m_LayerNodeStore->Return( release_node );
	      }
	
	    else if (new_value < LOWER_ACTIVE_THRESHOLD)
	      { // This index will move DOWN into a negative (inside) layer.
LOUT << "7new_value = " << new_value << "< LOWER_ACTIVE_THRESHOLD" << std::endl;
	      // First check for active layer neighbors moving in the opposite
	      // direction.
	      flag = false;
	      for (i = 0; i < m_NeighborList.GetSize(); ++i)
	        {
	        if (m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i))
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
	      
	      rms_change_accumulator += vnl_math_sqr(new_value - centerVal);	      
	      LOUT << "8rms_change_accumulator = " << rms_change_accumulator << std::endl;
	          
	      // Search the neighborhood for outside indicies.
	      temp_value = new_value + commonConf->m_ConstantGradientValue;
	      LOUT << "9temp_value = " << temp_value << std::endl;
	      for (i = 0; i < m_NeighborList.GetSize(); ++i)
	        {
	        idx = m_NeighborList.GetArrayIndex(i);
	        neighbor_status = m_statusIter.GetPixel( idx );
	        char tmp[2]; tmp[1] = 0;
	               tmp[0] = (neighbor_status + '0');
	        LOUT << "10status on " << idx << ":" << tmp << std::endl;
	        if (neighbor_status == 2)
	          {
	          // Keep the smallest magnitude value for this active set node.  This
	          // places the node closest to the active layer.
	        	ValueType pix = m_outIter.GetPixel(idx);
	        	LOUT << "11neighbor_status == 2, pix=" << pix << std::endl;
	          if ( pix >= UPPER_ACTIVE_THRESHOLD ||
	               ::vnl_math_abs(temp_value) < ::vnl_math_abs(pix) )
	            {
	            m_outIter.SetPixel(idx, temp_value);
	            }
	          }
	        }
	      node = m_LayerNodeStore->Borrow();
	      node->m_Value = layerIt->m_Value;
	      DownList->PushFront(node);
	      LOUT << "12DownList->PushFront(node)" << std::endl;
	      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingDown);
	
	      // Now remove this index from the active list.
	      release_node = layerIt.GetPointer();
	      ++layerIt;
	      m_Layers[0]->Unlink(release_node);
	      m_LayerNodeStore->Return( release_node );
	      }
	    else
	      {
	      rms_change_accumulator += vnl_math_sqr(new_value - centerVal);
	      LOUT << "13else: " << new_value << ", acc: " << rms_change_accumulator << std::endl;
	      //rms_change_accumulator += (*updateIt) * (*updateIt);
	      m_outIter.SetCenterPixel( new_value );
	      ++layerIt;
	      }
	    ++updateIt;
	    ++counter;
	    }
  
	  LOUT << std::endl << "14rms accum: " << rms_change_accumulator << "counter: " << counter << std::endl;
	
  // Determine the average change during this iteration.
  if (counter == 0)
    return this->m_ValueZero;
  else
  {
	  ValueType ret =
		  static_cast<double>( sqrt((double)(rms_change_accumulator / static_cast<ValueType>(counter)) ));
		  
		  LOUT << "returning: " << ret << std::endl;	  
		  return ret;
  }
}

///////////////////////////////////////////////////////////////////////////////
