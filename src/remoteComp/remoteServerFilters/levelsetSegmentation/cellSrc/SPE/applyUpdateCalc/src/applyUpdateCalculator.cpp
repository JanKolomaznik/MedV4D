#include "common/Types.h"
#include "../applyUpdateCalculator.h"
#include "../../vnl_math.h"
#include <string.h>

using namespace M4D::Cell;

#define DEBUG_ALG 12
#define DBG_LAYER_IT 12

//#define DEEP_DEBUG

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ApplyUpdateSPE(SharedResources *shaRes) :
	LayerValuesPropagator(shaRes), m_stepConfig(&shaRes->_changeConfig),
			m_localNodeStore(m_localNodeStoreBuffer),
			m_updateValuesIt(shaRes->_buf), m_ElapsedIterations(0)
{

}

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::~ApplyUpdateSPE()
{

}

///////////////////////////////////////////////////////////////////////////////

ApplyUpdateSPE::ValueType ApplyUpdateSPE::ApplyUpdate(TimeStepType dt)
{
	D_COMMAND(if(dt == 0) D_PRINT("WARNING: ApplyUpdate: dt=0"));

	MyLayerType UpList[2];
	MyLayerType DownList[2];

	//  LOG("Update list:");
	//  for(typename UpdateBufferType::iterator it = m_UpdateBuffer.begin(); it != m_UpdateBuffer.end(); it++)
	//	  LOUT << *it << ", ";
	//  LOG("");

	// prepare neighbour preloaders
	m_valueNeighPreloader.SetImageProps(&commonConf->valueImageProps);
	m_statusNeighPreloader.SetImageProps(&commonConf->statusImageProps);
	m_statusUpdatePreloader.SetImageProps(&commonConf->statusImageProps);

	// prepare iterator over update value list
	this->m_updateValuesIt.SetArray(m_stepConfig->updateBuffBegin);
	// prepare iterator over active layer
	this->m_layerIterator.SetBeginEnd(m_stepConfig->layer0Begin,
			m_stepConfig->layer0End);

	uint32 counter = 0;
	ValueType rms_change_accumulator = this->m_ValueZero;

	m_valueNeighPreloader.Reset();
	m_statusNeighPreloader.Reset();
	
#ifdef FOR_CELL
	m_statusUpdatePreloader.ReserveTags();
#endif

	if (m_layerIterator.IsLoading())
	{
		// pre-load first bunch of neighbs
		_loaded = m_layerIterator.GetLoaded();
		m_valueNeighPreloader.Load(*_loaded);
		m_statusNeighPreloader.Load(*_loaded);
	}
	
	//m_statusIter.GetNeighborhood().PrintImageToFile("statusbeforeUpdateActiveVals");
	//m_outIter.GetNeighborhood().PrintImageToFile("beforeUpdateActiveVals");

	while (m_valueNeighPreloader.GetCurrNodesNext() != m_stepConfig->layer0End)
	{
		// do one run
		UpdateActiveLayerValues(dt, &UpList[0], &DownList[0], counter,
				rms_change_accumulator);
		
		

		// Process the status up/down lists.  This is an iterative process which
		// proceeds outwards from the active layer.  Each iteration generates the
		// list for the next iteration.
		ProcessStatusLists(UpList, DownList);
	}
	//m_outIter.GetNeighborhood().PrintImageToFile("afterUpdateActiveVals");
#ifdef FOR_CELL
	DL_PRINT(DEBUG_ALG, "\n14rms accum: %f, counter: %u\n", rms_change_accumulator, counter);
	this->m_updateValuesIt.WaitForTransfer();
	
	m_statusUpdatePreloader.ReturnTags();
#else
	DL_PRINT(DEBUG_ALG, std::endl << "14rms accum: " << rms_change_accumulator << "counter: " << counter);
#endif
	
	//m_statusIter.GetNeighborhood().PrintImageToFile("statusafterUpdateActiveVals");

	// Finally, we update all of the layer values (excluding the active layer,
	// which has already been updated).
	this->PropagateAllLayerValues();

	m_ElapsedIterations++;

	// wait for ops to guarantee all is complete before this method ends
	// and to return its tags back to gate
	//	m_valueNeighPreloader.Fini();
	//	m_statusNeighPreloader.Fini();

#ifdef FOR_CELL
#else
	DL_PRINT(DEBUG_ALG, "returning: " << rms_change_accumulator);
#endif

	return rms_change_accumulator;
}

///////////////////////////////////////////////////////////////////////////////

void ApplyUpdateSPE::ProcessStatusLists(MyLayerType *UpLists,
		MyLayerType *DownLists)
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
		this->ProcessStatusList(&DownLists[j], &DownLists[k], down_to,
				down_search);

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
	this->ProcessStatusList(&DownLists[j], &DownLists[k], down_to,
			this->m_StatusNull);

	// Now we are left with the lists of indicies which must be
	// brought into the outermost layers.  Bring UpList into last inside layer
	// and DownList into last outside layer.
	this->ProcessOutsideList(&UpLists[k], static_cast<int>(LYERCOUNT) -2);
	this->ProcessOutsideList(&DownLists[k], static_cast<int>(LYERCOUNT) -1);
}

///////////////////////////////////////////////////////////////////////////////

void ApplyUpdateSPE::ProcessOutsideList(MyLayerType *OutsideList,
		StatusType ChangeToStatus)
{
	SparseFieldLevelSetNode *node;

	MyLayerType::Iterator it;
		
	m_statusUpdatePreloader.Reset();
	
	OutsideList->InitIterator(it);
	if(it.HasNext())
		m_statusUpdatePreloader.Load(*it.Next());

	// Push each index in the input list into its appropriate status layer
	// (ChangeToStatus) and update the status image value at that index.
	while ( !OutsideList->Empty() )
	{
#ifndef FOR_CELL
		DL_PRINT(DEBUG_ALG, "m_StatusImage->SetPixel(" << OutsideList->Front()->m_Value << ")=" << ((uint32)ChangeToStatus) );
#endif
		//m_statusIter.SetLocation(OutsideList->Front()->m_Value);
		m_statusIter.SetNeighbourhood(m_statusUpdatePreloader.GetLoaded());
		
		if(it.HasNext())
				m_statusUpdatePreloader.Load(*it.Next());
		
		m_statusIter.SetCenterPixel(ChangeToStatus);
		node = OutsideList->Front();
		OutsideList->PopFront();

#ifndef FOR_CELL
		DL_PRINT(DEBUG_ALG, "1: pop ," << OutsideList->Size() << " node " << node->m_Value);
#endif

		this->m_layerGate.PushToLayer(node, ChangeToStatus);
		m_localNodeStore.Return(node);
		
		m_statusUpdatePreloader.SaveCurrItem();
	}
}
///////////////////////////////////////////////////////////////////////////////
void ApplyUpdateSPE::ProcessStatusList(MyLayerType *InputList,
		MyLayerType *OutputList, StatusType ChangeToStatus,
		StatusType SearchForStatus)
{
	unsigned int i;
	bool bounds_status;
	SparseFieldLevelSetNode *node;
	StatusType neighbor_status;
	MyLayerType::Iterator it;
	
	m_statusUpdatePreloader.Reset();
	
	InputList->InitIterator(it);
	if(it.HasNext())
		m_statusUpdatePreloader.Load(*it.Next());

	// Push each index in the input list into its appropriate status layer
	// (ChangeToStatus) and update the status image value at that index.
	// Also examine the neighbors of the index to determine which need to go onto
	// the output list (search for SearchForStatus).
	while ( !InputList->Empty() )
	{
		m_statusIter.SetNeighbourhood(m_statusUpdatePreloader.GetLoaded());
		
		if(it.HasNext())
				m_statusUpdatePreloader.Load(*it.Next());
		
		node = InputList->Front();
		//D_PRINT("currnode" << "=" << node->m_Value);
//				<< " x nigb.node=" << m_statusIter.GetNeighborhood().m_currIndex);
		//m_statusIter.SetLocation(node->m_Value);
		
		m_statusIter.SetCenterPixel(ChangeToStatus);

#ifndef FOR_CELL
		DL_PRINT(DEBUG_ALG, "1. node=" << node->m_Value);
#endif

		InputList->PopFront(); // Must unlink from the input list  _before_ transferring to another list.

#ifndef FOR_CELL
		DL_PRINT(DEBUG_ALG, "2: pop ," << OutputList->Size() << " node " << node->m_Value);
#endif
		
		this->m_layerGate.PushToLayer(node, ChangeToStatus);
		// and return it to local store
		m_localNodeStore.Return(node);

		for (i = 0; i < m_NeighborList.GetSize(); ++i)
		{
			neighbor_status = m_statusIter.GetPixel(
					m_NeighborList.GetArrayIndex(i), bounds_status);

#ifndef FOR_CELL
			DL_PRINT(DEBUG_ALG, "2. neighbor_status=" << ((int32)neighbor_status) );
#endif

			if (neighbor_status == SearchForStatus)
			{ // mark this pixel so we don't add it twice. //TODO
				DL_PRINT(DEBUG_ALG, "3. neighbor_status == SearchForStatus");
				m_statusIter.SetPixel(m_NeighborList.GetNeighborhoodOffset(i),
						this->m_StatusChanging);
				if (bounds_status == true)
				{
					node = m_localNodeStore.Borrow();
					// reuse node from this loop coz it should be returned to local store
					node->m_Value = m_statusIter.GetIndex()
							+ m_NeighborList.GetNeighborhoodOffset(i);

#ifndef FOR_CELL
					DL_PRINT(DEBUG_ALG, "4. pushing to outList node: " << node->m_Value);

					DL_PRINT(DEBUG_ALG, "3: push," << OutputList->Size() << " node " << node->m_Value);
#endif
					OutputList->PushFront(node);

				} // else this index was out of bounds.
			}
		}
		
		m_statusUpdatePreloader.SaveCurrItem();
	}
}

///////////////////////////////////////////////////////////////////////////////

void ApplyUpdateSPE::UpdateActiveLayerValues(TimeStepType dt,
		MyLayerType *UpList, MyLayerType *DownList, uint32 &counter,
		ValueType &rms_change_accumulator)
{
	const ValueType LOWER_ACTIVE_THRESHOLD =
			- (commonConf->m_ConstantGradientValue / 2.0);
	const ValueType UPPER_ACTIVE_THRESHOLD =
			commonConf->m_ConstantGradientValue / 2.0;
	ValueType new_value, temp_value;
	StatusType neighbor_status;
	unsigned int i, idx;
	SparseFieldLevelSetNode *node;
	bool flag; //bounds_status, 

	ValueType centerVal;
	SparseFieldLevelSetNode *currNode;

#define MAX_TURN_LENGHT 2

	uint16 turnCounter = 0;

	while (m_valueNeighPreloader.GetCurrNodesNext() != m_stepConfig->layer0End
			&& turnCounter < MAX_TURN_LENGHT)
	{
		currNode = this->m_layerIterator.GetCurrItem();
		turnCounter ++;

#ifndef FOR_CELL
		DL_PRINT(DBG_LAYER_IT,
				"UpdateActiveLayerValues node: " << currNode->m_Value << "="
				<< (SparseFieldLevelSetNode *)this->m_layerIterator.GetCentralMemAddrrOfCurrProcessedNode().Get64());
#endif
		if (m_layerIterator.IsLoading())
		{
			// pre-load next neigborhoods
			_loaded = m_layerIterator.GetLoaded();
			m_valueNeighPreloader.Load(*_loaded);
			m_statusNeighPreloader.Load(*_loaded);
		}

		m_outIter.SetNeighbourhood(m_valueNeighPreloader.GetLoaded());
		m_statusIter.SetNeighbourhood(m_statusNeighPreloader.GetLoaded());
		
#ifdef DEEP_DEBUG
#ifdef FOR_CELL
		D_PRINT("node:[%d,%d,%d]\n", currNode->m_Value[0], currNode->m_Value[1], currNode->m_Value[2]);
#else
		D_PRINT("node:" << currNode->m_Value);
#endif
		m_statusIter.GetNeighborhood().Print();
		m_outIter.GetNeighborhood().Print();
#endif

		centerVal = m_outIter.GetCenterPixel();

		new_value = this->CalculateUpdateValue(dt, centerVal,
				m_updateValuesIt.GetCurrVal());

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
				++m_updateValuesIt;
				this->m_layerIterator.Next();
				// save to propagte changes in neighbourhoods
						m_valueNeighPreloader.SaveCurrItem();
						m_statusNeighPreloader.SaveCurrItem();
				continue;
			}

			rms_change_accumulator += vnl_math_sqr(new_value- centerVal);

			// Search the neighborhood for inside indicies.
			temp_value = new_value - commonConf->m_ConstantGradientValue;
			for (i = 0; i < m_NeighborList.GetSize(); ++i)
			{
				idx = m_NeighborList.GetArrayIndex(i);
				neighbor_status = m_statusIter.GetPixel(idx);
				char tmp[2];
				tmp[1] = 0;
				tmp[0] = (neighbor_status + '0');
				if (neighbor_status == 1)
				{
					// Keep the smallest possible value for the new active node.  This
					// places the new active layer node closest to the zero level-set.
					ValueType pix = m_outIter.GetPixel(idx);
					if (pix < LOWER_ACTIVE_THRESHOLD ||:: vnl_math_abs(temp_value) < ::vnl_math_abs(pix) )
					{
						m_outIter.SetPixel(idx, temp_value);
						//D_PRINT("Setting" << idx << "to" << temp_value);
					}
				}
			}
			node = m_localNodeStore.Borrow();
			node->m_Value = currNode->m_Value;

#ifndef FOR_CELL
		  DL_PRINT(DEBUG_ALG, "A1. pushing up node:" << node->m_Value);
	      DL_PRINT(DEBUG_ALG, "4: push," << UpList->Size() << " node " << node->m_Value);
#endif
	      UpList->PushFront(node);
	      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingUp);
	
	      // Now remove this index from the active list.
	      this->m_layerGate.UnlinkNode(
	    		  this->m_layerIterator.GetCentralMemAddrrOfCurrProcessedNode(), 0);
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
	        ++m_updateValuesIt;
	        this->m_layerIterator.Next();
	        // save to propagte changes in neighbourhoods
	        		m_valueNeighPreloader.SaveCurrItem();
	        		m_statusNeighPreloader.SaveCurrItem();
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
	            //D_PRINT("Setting" << idx << "to" << temp_value);
	            }
	          }
	        }
	      node = m_localNodeStore.Borrow();
	      node->m_Value = currNode->m_Value;
#ifndef FOR_CELL
	      DL_PRINT(DEBUG_ALG, "A2. pushing down node:" << node->m_Value );
	      
	      DL_PRINT(DEBUG_ALG, "5: push," << DownList->Size() << " node " << node->m_Value);
#endif
	      DownList->PushFront(node);
	      m_statusIter.SetCenterPixel(this->m_StatusActiveChangingDown);
	
	      // Now remove this index from the active list.
	      this->m_layerGate.UnlinkNode(
	    		  this->m_layerIterator.GetCentralMemAddrrOfCurrProcessedNode(), 0);
	      }
	    else
	      {
	      rms_change_accumulator += vnl_math_sqr(new_value - centerVal);
	      //rms_change_accumulator += (*updateIt) * (*updateIt);
	      //D_PRINT("Setting" << m_outIter.GetNeighborhood().m_currIndex << "to" << new_value);
	      m_outIter.SetCenterPixel( new_value );
	      }
	    ++m_updateValuesIt;
	    this->m_layerIterator.Next();
	    // save to propagte changes in neighbourhoods
	    		m_valueNeighPreloader.SaveCurrItem();
	    		m_statusNeighPreloader.SaveCurrItem();
	    ++counter;
	}	
}

///////////////////////////////////////////////////////////////////////////////
