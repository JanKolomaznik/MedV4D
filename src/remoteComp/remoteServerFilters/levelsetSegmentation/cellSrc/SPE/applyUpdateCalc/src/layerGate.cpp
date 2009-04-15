
#include "common/Types.h"
#include "common/Debug.h"
#include "../layerGate.h"

using namespace M4D::Cell;

#define DEBUG_GATE 12

///////////////////////////////////////////////////////////////////////////////

void LayerGate::UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	uint32 message = 0;
	message |= (UNLINKED_NODES_PROCESS & MessageID_MASK);
	message |= ((layerNum << MessageLyaerID_SHIFT) & MessageLyaerID_MASK);
	
	uint64 nodeAddress = (uint64) node;
	
	DL_PRINT(DEBUG_GATE, "Send ULNK, node:" << node << "layer: " << (uint32)layerNum);
	
#ifdef FOR_CELL
	// push to mailbox
#endif
#ifdef FOR_PC
	dispatcher->MyPushMessage(message);
	// push node address word by word
	dispatcher->MyPushMessage((uint32) (nodeAddress & 0xffffffff));
	dispatcher->MyPushMessage((uint32) (nodeAddress & ((uint64)0xffffffff << 32) ));
	
	// symulate dispatcher run
	message = dispatcher->MyPopMessage();
	dispatcher->DispatchMessage(message);
#endif
}

///////////////////////////////////////////////////////////////////////////////

//void LayerGate::ReturnToNodeStore(SparseFieldLevelSetNode *node)
//{
//	m_LayerNodeStore->Return(node);
//}

///////////////////////////////////////////////////////////////////////////////

void LayerGate::PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	uint32 message = 0;
	
	message |= (PUSHED_NODES_PROCESS & MessageID_MASK);
	message |= ((layerNum << MessageLyaerID_SHIFT) & MessageLyaerID_MASK);
	
	// convert node's value vector into the message param (24bits)
	// NOTE: vectors with values higher than 2^24 will be malformed
	// but since we transfer image coord and image are no larger that 1000
	// this limitation is painless
	uint32 param = node->m_Value[0];
	param |= (node->m_Value[1] << 8);
	param |= (node->m_Value[2] << 16);
	
	DL_PRINT(DEBUG_GATE, "Send PUSH, node:" << node->m_Value << "layer: " << (uint32)layerNum);
	
	message |= ((param << MessagePARAM_SHIFT) & MessagePARAM_MASK);
	
#ifdef FOR_PC
	dispatcher->MyPushMessage(message);
	
	// symulate dispatcher run
	message = dispatcher->MyPopMessage();
	dispatcher->DispatchMessage(message);
#endif
}

///////////////////////////////////////////////////////////////////////////////
