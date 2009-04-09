#ifndef APPLYUPDATECALCULATOR_H_
#define APPLYUPDATECALCULATOR_H_

#include "common/Types.h"
#include "../configStructures.h"
// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"

#include "../../supportClasses.h"
#include "../../commonConsts.h"

// to remove
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"

namespace M4D {
namespace Cell {

class ApplyUpdateSPE : public Consts
{
public:
	ApplyUpdateSPE();
	~ApplyUpdateSPE();
	
	void ApplyUpdate(TimeStepType dt);
	
	void PropagateAllLayerValues();
	
	ApplyUpdateConf conf;
	RunConfiguration *commonConf;
	
	void SetCommonConfiguration(RunConfiguration *c) { commonConf = c; }
	
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	//to remove
	typedef itk::SparseFieldLayer<SparseFieldLevelSetNode> LayerType;
	LayerType **m_Layers;
	typedef itk::ObjectStore<SparseFieldLevelSetNode> LayerNodeStorageType;
	LayerNodeStorageType *m_LayerNodeStore;
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
	                       StatusType promote, uint32 InOrOut);
	
	void UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum);
	void ReturnToNodeStore(SparseFieldLevelSetNode *node);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	
};

}
}
#endif /*APPLYUPDATECALCULATOR_H_*/
