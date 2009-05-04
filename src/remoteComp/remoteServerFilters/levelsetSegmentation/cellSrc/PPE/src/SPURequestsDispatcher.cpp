
#include "common/Common.h"
#include "../../SPE/commonTypes.h"
#include "../SPURequestsDispatcher.h"

using namespace M4D::Cell;
using namespace M4D::Multithreading;

#define DEBUG_MAILBOX 12

///////////////////////////////////////////////////////////////////////////////

void
SPURequestsDispatcher::DispatchMessage(uint32 message)
{
	MessageID id = (MessageID) (message & MessageID_MASK);	
	switch(id)
	{
	case UNLINKED_NODES_PROCESS:
		DispatchUnlinkMessage(message);
		break;
	case PUSHED_NODES_PROCESS:
		DispatchPushNodeMess(message);
		break;
	}
}
///////////////////////////////////////////////////////////////////////////////
void
SPURequestsDispatcher::DispatchPushNodeMess(uint32 message)
{
	
	uint8 lyerID = (message & MessageLyaerID_MASK) >> MessageLyaerID_SHIFT;
	uint32 param = (message & MessagePARAM_MASK) >> MessagePARAM_SHIFT;
	
	uint32 secondWord = MyPopMessage();
	
	TIndex ind = { {
			param,
			secondWord & 0xFFFF,
			(secondWord & (0xFFFF << 16)) >> 16
	} };
	
	_workManager->PUSHNode(ind, lyerID);
	DL_PRINT(DEBUG_MAILBOX, "PUSH " << ind << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////
void
SPURequestsDispatcher::DispatchUnlinkMessage(uint32 message)
{
	uint8 lyerID = (message & MessageLyaerID_MASK) >> MessageLyaerID_SHIFT;
	
	uint64 nodeAddress = MyPopMessage();
	nodeAddress |= ((uint64) MyPopMessage()) << 32;
	
	LayerNodeType *n = (LayerNodeType *) nodeAddress;
	
	_workManager->UNLINKNode(n, lyerID);
	DL_PRINT(DEBUG_MAILBOX, "ULNK " << n << " layr=" << (uint32)lyerID);
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_PC
void
SPURequestsDispatcher::MyPushMessage(uint32 message)
{
	ScopedLock lock(mutex);
	messageQueue.push(message);
}

///////////////////////////////////////////////////////////////////////////////

uint32
SPURequestsDispatcher::MyPopMessage()
{
	ScopedLock lock(mutex);
	uint32 val = messageQueue.front();
	messageQueue.pop();
	return val;
}
#endif
///////////////////////////////////////////////////////////////////////////////
