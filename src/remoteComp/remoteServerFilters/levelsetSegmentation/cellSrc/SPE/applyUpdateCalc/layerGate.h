#ifndef LAYERGATE_H_
#define LAYERGATE_H_

#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
#define FOR_CELL
#else
#define FOR_PC
#endif

#include "../commonTypes.h"
//#include "../configStructures.h"
//#include "../tools/cellRemoteArray.h"

#ifdef FOR_PC
#include "common/Common.h"
#include "../../PPE/SPURequestsDispatcher.h"
#endif

namespace M4D {
namespace Cell {

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
