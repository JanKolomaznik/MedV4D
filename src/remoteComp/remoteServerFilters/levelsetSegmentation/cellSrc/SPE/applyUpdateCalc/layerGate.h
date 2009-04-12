#ifndef LAYERGATE_H_
#define LAYERGATE_H_

#include "../configStructures.h"

// to remove
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"

namespace M4D {
namespace Cell {

class LayerGate
{
public:
	
	//to remove
	typedef itk::SparseFieldLayer<SparseFieldLevelSetNode> LayerType;
	typedef LayerType::Pointer LayerPointerType;
	typedef itk::ObjectStore<SparseFieldLevelSetNode> LayerNodeStorageType;
	
	void UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum);
	void ReturnToNodeStore(SparseFieldLevelSetNode *node);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
	
	
	LayerType **m_Layers;
	LayerNodeStorageType *m_LayerNodeStore;			
		
};

}
}
#endif /*LAYERGATE_H_*/
