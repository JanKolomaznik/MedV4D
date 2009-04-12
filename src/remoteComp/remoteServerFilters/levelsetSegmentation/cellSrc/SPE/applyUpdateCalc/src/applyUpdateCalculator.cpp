#include "common/Types.h"
#include "common/Common.h"
#include "../applyUpdateCalculator.h"
#include "../../vnl_math.h"
#include <math.h>	//sqrt
#include <string.h>
#include <fstream>
#include <iostream>

using namespace M4D::Cell;

#define DEBUG_ALG 12

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ApplyUpdateSPE()
	: m_ElapsedIterations(0)
{

}

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::~ApplyUpdateSPE()
{

}

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ValueType
ApplyUpdateSPE::ApplyUpdate(TimeStepType dt)
{
	MyLayerType UpList[2];
	MyLayerType DownList[2];
	
	DL_PRINT(DEBUG_ALG, "ApplyUpdate" << std::endl);

	//  LOG("Update list:");
	//  for(typename UpdateBufferType::iterator it = m_UpdateBuffer.begin(); it != m_UpdateBuffer.end(); it++)
	//	  LOUT << *it << ", ";
	//  LOG("");
	
	NeighborhoodCell<TPixelValue> outNeigh;
	outNeigh.SetImageProperties( &commonConf->valueImageProps);
	NeighborhoodCell<StatusType> statusNeigh;
	statusNeigh.SetImageProperties( &commonConf->statusImageProps);

	m_outIter.SetNeighbourhood( &outNeigh);
	m_statusIter.SetNeighbourhood( &statusNeigh);
	
	// prepare iterator over update value list
	this->m_updateValuesIt.SetArray(commonConf->m_UpdateBufferData);	
	// prepare iterator over active layer
	this->m_layerIterator.SetBeginEnd(conf.layerBegins[0], conf.layerEnds[0]);
		
	uint32 counter = 0;
	ValueType rms_change_accumulator = this->m_ValueZero;
		
	while(m_layerIterator.HasNext())  
	{
		// do one run
		UpdateActiveLayerValues(dt, &UpList[0], &DownList[0], counter, rms_change_accumulator);

	// Process the status up/down lists.  This is an iterative process which
	// proceeds outwards from the active layer.  Each iteration generates the
	// list for the next iteration.
	
//	  std::stringstream s;
//	  s << "before" << this->m_ElapsedIterations;
//	  std::ofstream b(s.str().c_str());
//	m_statusIter.GetNeighborhood().PrintImage(b);

		ProcessStatusLists(UpList, DownList);
	}
	
	  DL_PRINT(DEBUG_ALG, std::endl << "14rms accum: " << rms_change_accumulator << "counter: " << counter);
		
	  ValueType retval;
	  
	  // Determine the average change during this iteration.
	  if (counter == 0)
		  retval = this->m_ValueZero;
	  else
	  {
		  retval =  static_cast<double>( 
				  sqrt((double)(rms_change_accumulator / static_cast<ValueType>(counter)) ));			  
	  }
	
//	std::stringstream s3;
//		  s3 << "afterOutside" << this->m_ElapsedIterations;
	//	  std::ofstream a1(s3.str().c_str());
	//m_statusIter.GetNeighborhood().PrintImage(a1);

	// Finally, we update all of the layer values (excluding the active layer,
	// which has already been updated).
	this->PropagateAllLayerValues();
	
	m_ElapsedIterations++;
	
	//LOUT << "returning: " << ret << std::endl;	
	return retval;
}

///////////////////////////////////////////////////////////////////////////////

void
ApplyUpdateSPE::ProcessStatusLists(
		MyLayerType *UpLists, MyLayerType *DownLists)
{
	unsigned int j, k, t;

	StatusType up_to, up_search;
	StatusType down_to, down_search;
	
	// First process the status lists generated on the active layer.
	this->ProcessStatusList(&UpLists[0], &UpLists[1], 2, 1);
	this->ProcessStatusList(&DownLists[0], &DownLists[1], 1, 2);

	down_to = up_to = 0;
	up_search = 3;
	down_search = 4;
	j = 1;
	k = 0;
	while (down_search < static_cast<StatusType>( LYERCOUNT ) )
	{
		this->ProcessStatusList(&UpLists[j], &UpLists[k], up_to, up_search);
		this->ProcessStatusList(&DownLists[j], &DownLists[k], down_to, down_search);

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
	this->ProcessStatusList(&UpLists[j], &UpLists[k], up_to, this->m_StatusNull);
	this->ProcessStatusList(&DownLists[j], &DownLists[k], down_to, this->m_StatusNull);
	
//	 std::stringstream s2;
//			  s2 << "beforeOutside" << this->m_ElapsedIterations;
//			  std::ofstream b1(s2.str().c_str());
//	m_statusIter.GetNeighborhood().PrintImage(b1);

	// Now we are left with the lists of indicies which must be
	// brought into the outermost layers.  Bring UpList into last inside layer
	// and DownList into last outside layer.
	this->ProcessOutsideList(&UpLists[k], static_cast<int>(LYERCOUNT) -2);
	this->ProcessOutsideList(&DownLists[k], static_cast<int>(LYERCOUNT) -1);
}

///////////////////////////////////////////////////////////////////////////////

void
ApplyUpdateSPE::ProcessOutsideList(
		MyLayerType *OutsideList, StatusType ChangeToStatus)
{
	SparseFieldLevelSetNode *node;
	
	//LOUT << "ProcessOutsideList" << std::endl << std::endl;

	// Push each index in the input list into its appropriate status layer
	// (ChangeToStatus) and update the status image value at that index.
	while ( !OutsideList->Empty() )
	{
		DL_PRINT(DEBUG_ALG, "m_StatusImage->SetPixel(" << OutsideList->Front()->m_Value << ")=" << ((uint32)ChangeToStatus) );
		m_statusIter.SetLocation(OutsideList->Front()->m_Value);
		m_statusIter.SetCenterPixel(ChangeToStatus);
		node = OutsideList->Front();
		
		OutsideList->PopFront();
		LOG("1: pop ," << OutsideList->Size() << " node " << node->m_Value);
		
		this->m_layerGate.PushToLayer(node, ChangeToStatus);
		
		m_localNodeStore.Return(node);
//		m_Layers[ChangeToStatus]->PushFront(node);
	}
}
///////////////////////////////////////////////////////////////////////////////
void
ApplyUpdateSPE::ProcessStatusList(
		MyLayerType *InputList, MyLayerType *OutputList, StatusType ChangeToStatus,
		StatusType SearchForStatus)
{
	unsigned int i;
	bool bounds_status;
	SparseFieldLevelSetNode *node;
	StatusType neighbor_status;
//	NeighborhoodIterator<StatusImageType> statusIt(m_NeighborList.GetRadius(),
//			m_StatusImage, this->GetOutput()->GetRequestedRegion());
		
	DL_PRINT(DEBUG_ALG, "ProcessStatusList" << std::endl);

	// Push each index in the input list into its appropriate status layer
	// (ChangeToStatus) and update the status image value at that index.
	// Also examine the neighbors of the index to determine which need to go onto
	// the output list (search for SearchForStatus).
	while ( !InputList->Empty() )
	{
		node = InputList->Front();
		m_statusIter.SetLocation(node->m_Value);		
		m_statusIter.SetCenterPixel(ChangeToStatus);

		DL_PRINT(DEBUG_ALG, "1. node=" << node->m_Value);
		
		
		InputList->PopFront(); // Must unlink from the input list  _before_ transferring to another list.
		LOG("2: pop ," << OutputList->Size() << " node " << node->m_Value);
		
		//m_Layers[ChangeToStatus]->PushFront(node);
		this->m_layerGate.PushToLayer(node, ChangeToStatus);
		// and return it to local store
		m_localNodeStore.Return(node);

		for (i = 0; i < m_NeighborList.GetSize(); ++i)
		{
			//std::cout << "predIncriminovanym:" << std::endl << m_statusIter.GetNeighborhood() << std::endl;
			neighbor_status
					= m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i), bounds_status);
			DL_PRINT(DEBUG_ALG, "2. neighbor_status=" << ((uint32)neighbor_status) );

			if (neighbor_status == SearchForStatus)
			{ // mark this pixel so we don't add it twice. //TODO
				DL_PRINT(DEBUG_ALG, "3. neighbor_status == SearchForStatus");
				m_statusIter.SetPixel(m_NeighborList.GetNeighborhoodOffset(i),
						this->m_StatusChanging);
				if (bounds_status == true)
				{
					node = m_localNodeStore.Borrow();
					//node = BorrowFromLocalNodeStore();
					
					// reuse node from this loop coz it should be returned to local store
					node->m_Value = m_statusIter.GetIndex()
							+ m_NeighborList.GetNeighborhoodOffset(i);
					DL_PRINT(DEBUG_ALG, "4. pushing to outList node: " << node->m_Value);					
		
					LOG("3: push," << OutputList->Size() << " node " << node->m_Value);
					OutputList->PushFront(node);
					
				} // else this index was out of bounds.
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

void
ApplyUpdateSPE::UpdateActiveLayerValues(
		TimeStepType dt, MyLayerType *UpList, MyLayerType *DownList, 
		uint32 &counter, ValueType &rms_change_accumulator)
{	
	  const ValueType LOWER_ACTIVE_THRESHOLD = - (commonConf->m_ConstantGradientValue / 2.0);
	  const ValueType UPPER_ACTIVE_THRESHOLD =    commonConf->m_ConstantGradientValue / 2.0;
	  ValueType new_value, temp_value;
	  StatusType neighbor_status;
	  unsigned int i, idx;
	  SparseFieldLevelSetNode *node;
	  bool flag; //bounds_status, 
	  
	  ValueType centerVal;
	  SparseFieldLevelSetNode *currNode;	  
	  
#define MAX_TURN_LENGHT 2
	
	uint16 turnCounter = 0;
	  
		while (this->m_layerIterator.HasNext() && turnCounter < MAX_TURN_LENGHT)
		{
			currNode = this->m_layerIterator.Next();
			turnCounter ++;

		  m_outIter.SetLocation(currNode->m_Value);
		  m_statusIter.SetLocation(currNode->m_Value);
		  
		  centerVal = m_outIter.GetCenterPixel();
	
	    new_value = this->CalculateUpdateValue(dt, centerVal, m_updateValuesIt.GetCurrVal());
	
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
	        if (m_statusIter.GetPixel(m_NeighborList.GetArrayIndex(i))
	            == this->m_StatusActiveChangingDown)
	          {
	          flag = true;
	          break;
	          }
	        }
	      if (flag == true)
	        {
	        //++layerIt;
	        ++m_updateValuesIt;
	        continue;
	        }
	
	      rms_change_accumulator += vnl_math_sqr(new_value- centerVal );
	
	      // Search the neighborhood for inside indicies.
	      temp_value = new_value - commonConf->m_ConstantGradientValue;
	      for (i = 0; i < m_NeighborList.GetSize(); ++i)
	        {
	        idx = m_NeighborList.GetArrayIndex(i);
	        neighbor_status = m_statusIter.GetPixel( idx );
	        char tmp[2]; tmp[1] = 0;
	               tmp[0] = (neighbor_status + '0');
	        if (neighbor_status == 1)
	          {
	          // Keep the smallest possible value for the new active node.  This
	          // places the new active layer node closest to the zero level-set.
	        	ValueType pix = m_outIter.GetPixel(idx);
	          if ( pix < LOWER_ACTIVE_THRESHOLD ||
	               ::vnl_math_abs(temp_value) < ::vnl_math_abs(pix) )
	            {
	            m_outIter.SetPixel(idx, temp_value);
	            }
	          }
	        }
//	      node = m_LayerNodeStore->Borrow();
	      //node = BorrowFromLocalNodeStore();
	      node = m_localNodeStore.Borrow();
	      node->m_Value = currNode->m_Value;
	      DL_PRINT(DEBUG_ALG, "A1. pushing up node:" << node->m_Value);
	      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	      // tady se do uplistu da adresa z linked chain iteratoru, takze je treba 
	      // si predtim vzit z local objetStore
	      LOG("4: push," << UpList->Size() << " node " << node->m_Value);
	      UpList->PushFront(node);
	      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingUp);
	
	      // Now remove this index from the active list.
	      //release_node = layerIt.GetPointer();
	      //++layerIt;
	      //m_Layers[0]->Unlink(release_node);
	      this->m_layerGate.UnlinkNode(currNode, 0);
	      //m_LayerNodeStore->Return( release_node );
	      //this->m_layerGate.ReturnToNodeStore(currNode);
	      }
	
	    else if (new_value < LOWER_ACTIVE_THRESHOLD)
	      { // This index will move DOWN into a negative (inside) layer.
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
	        //++layerIt;
	        ++m_updateValuesIt;
	        continue;
	        }
	      
	      rms_change_accumulator += vnl_math_sqr(new_value - centerVal);
	          
	      // Search the neighborhood for outside indicies.
	      temp_value = new_value + commonConf->m_ConstantGradientValue;
	      for (i = 0; i < m_NeighborList.GetSize(); ++i)
	        {
	        idx = m_NeighborList.GetArrayIndex(i);
	        neighbor_status = m_statusIter.GetPixel( idx );
	        char tmp[2]; tmp[1] = 0;
	               tmp[0] = (neighbor_status + '0');
	        if (neighbor_status == 2)
	          {
	          // Keep the smallest magnitude value for this active set node.  This
	          // places the node closest to the active layer.
	        	ValueType pix = m_outIter.GetPixel(idx);
	          if ( pix >= UPPER_ACTIVE_THRESHOLD ||
	               ::vnl_math_abs(temp_value) < ::vnl_math_abs(pix) )
	            {
	            m_outIter.SetPixel(idx, temp_value);
	            }
	          }
	        }
//	      node = m_LayerNodeStore->Borrow();
	      //node = BorrowFromLocalNodeStore();
	      node = m_localNodeStore.Borrow();
	      node->m_Value = currNode->m_Value;
	      DL_PRINT(DEBUG_ALG, "A2. pushing down node:" << node->m_Value );
	      
	      LOG("5: push," << DownList->Size() << " node " << node->m_Value);
	      DownList->PushFront(node);
	      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingDown);
	
	      // Now remove this index from the active list.
	      //release_node = layerIt.GetPointer();
	      //++layerIt;
	      //m_Layers[0]->Unlink(release_node);
	      this->m_layerGate.UnlinkNode(currNode, 0);
//	      m_LayerNodeStore->Return( release_node );
	      //this->m_layerGate.ReturnToNodeStore(currNode);
	      }
	    else
	      {
	      rms_change_accumulator += vnl_math_sqr(new_value - centerVal);
	      //rms_change_accumulator += (*updateIt) * (*updateIt);
	      m_outIter.SetCenterPixel( new_value );
	      //++layerIt;
	      }
	    ++m_updateValuesIt;
	    ++counter;
	    }
  
	
}

///////////////////////////////////////////////////////////////////////////////
