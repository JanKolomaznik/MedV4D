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
	void SetCommonConfiguration(RunConfiguration *c) { commonConf = c; }
	
	//to remove
			typedef itk::SparseFieldLayer<SparseFieldLevelSetNode> LayerType;
			typedef LayerType::Pointer LayerPointerType;
			
			LayerType **m_Layers;
			typedef itk::ObjectStore<SparseFieldLevelSetNode> LayerNodeStorageType;
			LayerNodeStorageType *m_LayerNodeStore;
	
protected:
			
			RunConfiguration *commonConf;
			
			
			void UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum);
			void ReturnToNodeStore(SparseFieldLevelSetNode *node);
			void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
};

}
}
#endif /*LAYERGATE_H_*/
