#ifndef LAYERGATE_H_
#define LAYERGATE_H_

#include "../commonTypes.h"
//#include "../configStructures.h"
//#include "../tools/cellRemoteArray.h"

#ifdef FOR_PC
#include "common/Common.h"
#endif

namespace M4D {
namespace Cell {

#ifdef FOR_PC
class SPURequestsDispatcher;
#endif

class LayerGate
{
public:
	
#ifdef FOR_PC
	SPURequestsDispatcher *dispatcher;
#endif
	
//	typedef PUTRemoteArrayCell<TIndex, 8> TPutNodeArray;
	
	
	void UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum);
	//void ReturnToNodeStore(SparseFieldLevelSetNode *node);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
	
//	TPutNodeArray putArrays[LYERCOUNT];
//	TPutNodeArray unlinkArrays[LYERCOUNT];
};

}
}
#endif /*LAYERGATE_H_*/
