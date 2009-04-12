#ifndef LAYERVALSPROPAGATOR_H_
#define LAYERVALSPROPAGATOR_H_

#include "../../commonConsts.h"

// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"
#include "../tools/sparesFieldLayer.h"

#include "../../supportClasses.h"

#include "layerGate.h"

namespace M4D {
namespace Cell {

class LayerValuesPropagator : public Consts
{
public:
	void PropagateAllLayerValues();	
	
	void SetCommonConfiguration(RunConfiguration *c) { commonConf = c; }
			
	ApplyUpdateConf conf;
	
	
	void SetGateProps(LayerGate::LayerType **layers, LayerGate::LayerNodeStorageType *layerNodeStore)
	{
		m_layerGate.m_Layers = layers;
		m_layerGate.m_LayerNodeStore = layerNodeStore;
	}
	
protected:
	
	typedef LinkedChainIteratorCellWithLayerAccess<SparseFieldLevelSetNode> TLayerIterator;
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;
	
	RunConfiguration *commonConf;
	
	LayerGate m_layerGate;
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
				                       StatusType promote, uint32 InOrOut);
};

}
}
#endif /*LAYERVALSPROPAGATOR_H_*/
