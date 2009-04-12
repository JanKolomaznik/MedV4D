
#include "common/Types.h"
#include "../layerGate.h"

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////

void LayerGate::UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	m_Layers[layerNum]->Unlink(node);
	m_LayerNodeStore->Return(node);
}

///////////////////////////////////////////////////////////////////////////////

//void LayerGate::ReturnToNodeStore(SparseFieldLevelSetNode *node)
//{
//	m_LayerNodeStore->Return(node);
//}

///////////////////////////////////////////////////////////////////////////////

void LayerGate::PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum)
{
	SparseFieldLevelSetNode *tmp = m_LayerNodeStore->Borrow();
	tmp->m_Value = node->m_Value;
	m_Layers[layerNum]->PushFront(tmp);
}

///////////////////////////////////////////////////////////////////////////////