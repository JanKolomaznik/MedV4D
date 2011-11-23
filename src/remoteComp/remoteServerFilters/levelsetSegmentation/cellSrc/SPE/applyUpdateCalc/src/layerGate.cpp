
#include "MedV4D/Common/Types.h"
#include "../layerGate.h"

#ifdef FOR_CELL
#include <spu_mfcio.h>
#include "../../tools/SPEdebug.h"
#endif

using namespace M4D::Cell;

#define DEBUG_GATE 12

///////////////////////////////////////////////////////////////////////////////

void LayerGate::UnlinkNode(Address node, uint8 layerNum)
{
	uint32 message = 0;
	message |= (UNLINKED_NODES_PROCESS & MessageID_MASK);
	message |= ((layerNum << MessageLyaerID_SHIFT) & MessageLyaerID_MASK);
	
#ifdef FOR_CELL
	DL_PRINT(DEBUG_GATE, "Send ULNK, node: %lld layer: %d",
			node.Get64(), (uint32)layerNum);
#else
	DL_PRINT(DEBUG_GATE, "Send ULNK, node:" << (void*) node.Get64() 
			<< " layer: " << (uint32)layerNum);
#endif
	
#ifdef FOR_CELL
	// push to mailbox
	spu_writech(SPU_WrOutMbox, message);
	spu_writech(SPU_WrOutMbox, (uint32) (node.Get64() & 0xffffffff));
	spu_writech(SPU_WrOutMbox, (uint32) (node.Get64() >> 32));
#else
	{
		ScopedLock lock(_mailbox->fromSPEQMutex);
		
		_mailbox->SPEPush(message);
		// push node address word by word
		_mailbox->SPEPush((uint32) (node.Get64() & 0xffffffff));
		_mailbox->SPEPush((uint32) (node.Get64() >> 32));
	}
#endif
}

///////////////////////////////////////////////////////////////////////////////

void LayerGate::PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	uint32 message = 0;
	
	message |= (PUSHED_NODES_PROCESS & MessageID_MASK);
	message |= ((layerNum << MessageLyaerID_SHIFT) & MessageLyaerID_MASK);
	
#ifdef FOR_CELL
	DL_PRINT(DEBUG_GATE, "Send PUSH, node: %lld, layer: %d",
			(uint64) node, (uint32)layerNum);
#else
	DL_PRINT(DEBUG_GATE, "Send PUSH, node:" << node->m_Value << 
			" layer: " << (uint32)layerNum);
#endif
	
	// 1st coord will be passed with message
	// the two rest along the next word
	// NOTE: vectors with coords values higher than 2^16 will be malformed
	// but since we transfer image coord and image are no larger that 1000
	// this limitation is painless
	uint32 param = node->m_Value[0];	
	message |= ((param << MessagePARAM_SHIFT) & MessagePARAM_MASK);

#ifdef FOR_CELL
	spu_writech(SPU_WrOutMbox, message);
	
	message = (node->m_Value[1]);
	message |= (node->m_Value[2] << 16);
	
	spu_writech(SPU_WrOutMbox, message);
#else
	{
			ScopedLock lock(_mailbox->fromSPEQMutex);
			_mailbox->SPEPush(message);
			
			message = (node->m_Value[1]);
			message |= (node->m_Value[2] << 16);
			
			_mailbox->SPEPush(message);
	}
#endif	
}

///////////////////////////////////////////////////////////////////////////////
