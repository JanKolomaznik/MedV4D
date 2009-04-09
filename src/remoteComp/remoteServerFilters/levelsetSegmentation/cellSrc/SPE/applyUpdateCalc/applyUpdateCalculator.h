#ifndef APPLYUPDATECALCULATOR_H_
#define APPLYUPDATECALCULATOR_H_

#include "common/Types.h"
#include "../configStructures.h"
// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"

namespace M4D {
namespace Cell {

class ApplyUpdateSPE
{
public:
	void ApplyUpdate(TimeStepType dt);
	
	void PropagateAllLayerValues();
	
	ApplyUpdateConf &conf;
	RunConfiguration &commonConf;
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
	                       StatusType promote, uint32 InOrOut);
	
	void UnlinkNode(SparseFieldLevelSetNode *node);
	
	SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > *m_NeighborList;
	
	NeighbourIteratorCell m_outIter;
	NeighbourIteratorCell m_statusIter;
};

}
}
#endif /*APPLYUPDATECALCULATOR_H_*/
