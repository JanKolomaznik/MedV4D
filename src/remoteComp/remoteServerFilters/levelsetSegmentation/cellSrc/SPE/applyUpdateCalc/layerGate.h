#ifndef LAYERGATE_H_
#define LAYERGATE_H_

#include "../commonTypes.h"
//#include "../configStructures.h"
//#include "../tools/cellRemoteArray.h"

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
	
//	typedef PUTRemoteArrayCell<TIndex, 8> TPutNodeArray;
	
	
	void UnlinkNode(Address node, uint8 layerNum);
	//void ReturnToNodeStore(SparseFieldLevelSetNode *node);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
	
//	TPutNodeArray putArrays[LYERCOUNT];
//	TPutNodeArray unlinkArrays[LYERCOUNT];
};

}
}
#endif /*LAYERGATE_H_*/
