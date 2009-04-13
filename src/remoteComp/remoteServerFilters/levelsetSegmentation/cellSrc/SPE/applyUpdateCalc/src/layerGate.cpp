
#include "common/Types.h"
#include "../layerGate.h"

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////

void LayerGate::UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum)
{
//	m_Layers[layerNum]->Unlink(node);
//	m_LayerNodeStore->Return(node);
//	unlinkArrays[layerNum].push_back(node->m_value);
	uint32 message = 0;
	message |= (UNLINKED_NODES_PROCESS & MessageID_MASK);
	message |= (layerNum & MessageLyaerID_MASK);
	
	uint64 nodeAddress = (uint64) node;
	
	LOG("Send ULNK, node:" << node);
	
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
//	SparseFieldLevelSetNode *tmp = m_LayerNodeStore->Borrow();
//	tmp->m_Value = node->m_Value;
//	m_Layers[layerNum]->PushFront(tmp);
//	putArrays[layerNum].push_back(node->m_value);
	uint32 message = 0;
//	uint32 mask = MessageLyaerID_MASK;
//	uint32 pp = (layerNum & MessageLyaerID_MASK);
//	uint32 ee = MessageID_MASK;
//	uint32 ii = MessageID_MASK << 3;
	
	message |= (PUSHED_NODES_PROCESS & MessageID_MASK);
	message |= ((layerNum << MessageLyaerID_SHIFT) & MessageLyaerID_MASK);
	
	// convert node's value vector into the message param (24bits)
	// NOTE: vectors with values higher than 2^24 will be malformed
	// but since we transfer image coord and image are no larger that 1000
	// this limitation is painless
	uint32 param = node->m_Value[0];
	param |= (node->m_Value[1] << 8);
	param |= (node->m_Value[2] << 16);
	
	LOG("Send PUSH, node:" << node->m_Value);
	
//	uint32 oo = ~(ee | ii);
	message |= ((param << MessagePARAM_SHIFT) & MessagePARAM_MASK);
	
#ifdef FOR_PC
	dispatcher->MyPushMessage(message);
	
	// symulate dispatcher run
	message = dispatcher->MyPopMessage();
	dispatcher->DispatchMessage(message);
#endif
}

///////////////////////////////////////////////////////////////////////////////