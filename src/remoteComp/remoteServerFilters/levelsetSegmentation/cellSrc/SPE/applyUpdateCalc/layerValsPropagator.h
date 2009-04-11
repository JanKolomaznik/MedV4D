#ifndef LAYERVALSPROPAGATOR_H_
#define LAYERVALSPROPAGATOR_H_

#include "../../commonConsts.h"
#include "../configStructures.h"
// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"
#include "../tools/sparesFieldLayer.h"

#include "../../supportClasses.h"



// to remove
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"

namespace M4D {
namespace Cell {

class LayerValuesPropagator : public Consts
{
public:
	void PropagateAllLayerValues();	
	void SetCommonConfiguration(RunConfiguration *c) { commonConf = c; }
	
	
	//to remove
			typedef itk::SparseFieldLayer<SparseFieldLevelSetNode> LayerType;
			typedef LayerType::Pointer LayerPointerType;
			
			LayerType **m_Layers;
			typedef itk::ObjectStore<SparseFieldLevelSetNode> LayerNodeStorageType;
			LayerNodeStorageType *m_LayerNodeStore;
			
			
			ApplyUpdateConf conf;
	
protected:
	
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	RunConfiguration *commonConf;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;
	
	
	
	
	void UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum);
	void ReturnToNodeStore(SparseFieldLevelSetNode *node);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
				                       StatusType promote, uint32 InOrOut);
};

}
}
#endif /*LAYERVALSPROPAGATOR_H_*/
