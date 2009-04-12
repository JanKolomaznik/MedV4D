#ifndef LAYERVALSPROPAGATOR_H_
#define LAYERVALSPROPAGATOR_H_

#include "../../commonConsts.h"

// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"
#include "../tools/sparesFieldLayer.h"
#include "../tools/preloadedNeighbourhoods.h"

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
	
	LayerValuesPropagator();
	
	typedef LinkedChainIteratorCellWithLayerAccess<SparseFieldLevelSetNode> TLayerIterator;
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	typedef NeighborhoodCell<TPixelValue> TValueNeighborhood;
	typedef NeighborhoodCell<StatusType> TStatusNeighborhood;
	typedef PreloadedNeigborhoods<TValueNeighborhood, 4> TValueNeighbPreloadeder;
	typedef PreloadedNeigborhoods<TStatusNeighborhood, 4> TStatusNeighbPreloadeder;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;
	
	RunConfiguration *commonConf;
	
	LayerGate m_layerGate;
	TLayerIterator m_layerIterator;
	
	TValueNeighbPreloadeder m_valueNeighPreloader;
	TStatusNeighbPreloadeder m_statusNeighPreloader;
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
				                       StatusType promote, uint32 InOrOut);
};

}
}
#endif /*LAYERVALSPROPAGATOR_H_*/
