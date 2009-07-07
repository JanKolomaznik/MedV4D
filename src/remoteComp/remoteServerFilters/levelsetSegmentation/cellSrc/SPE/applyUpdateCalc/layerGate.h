#ifndef LAYERGATE_H_
#define LAYERGATE_H_

#include "../commonTypes.h"

#ifdef FOR_PC
#include "common/Common.h"
#include "../../PPE/mailboxSimulator.h"
#endif

namespace M4D {
namespace Cell {


class LayerGate
{
public:
	
#ifdef FOR_PC
	MailboxSimulator *_mailbox;
#endif	
	
	void UnlinkNode(Address node, uint8 layerNum);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
};

}
}
#endif /*LAYERGATE_H_*/
