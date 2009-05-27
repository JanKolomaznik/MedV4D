#ifndef LAYERVALSPROPAGATOR_H_
#define LAYERVALSPROPAGATOR_H_

#include "../../commonConsts.h"
#include "../configStructures.h"

#include "layerGate.h"

// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"
#include "../tools/sparesFieldLayer.h"
#include "../tools/preloadedNeighbourhoods.h"

#include "../../supportClasses.h"
#include "../tools/sharedResources.h"


namespace M4D {
namespace Cell {

class LayerValuesPropagator : public Consts
{
public:
	
	void PropagateAllLayerValues();
	
	LayerGate m_layerGate;
	
protected:
	
	LayerValuesPropagator(SharedResources *shaRes);
	
	typedef LinkedChainIteratorCell<SparseFieldLevelSetNode> TLayerIterator;
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	typedef NeighborhoodCell<TPixelValue> TValueNeighborhood;
	typedef NeighborhoodCell<StatusType> TStatusNeighborhood;
	typedef PreloadedNeigborhoods<TPixelValue, 3> TValueNeighbPreloadeder;
	typedef PreloadedNeigborhoods<StatusType, 3> TStatusNeighbPreloadeder;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;	
	
	TLayerIterator m_layerIterator;
	
	TValueNeighbPreloadeder m_valueNeighPreloader;
	TStatusNeighbPreloadeder m_statusNeighPreloader;
	
	PropagateValuesConf *m_propLayerValuesConfig;
	RunConfiguration *commonConf;
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
				                       StatusType promote);
	
	void DoTheWork(SparseFieldLevelSetNode *currNode, StatusType from, StatusType to,
			StatusType promote);
	
	TPixelValue value, value_temp, _delta;
};

}
}
#endif /*LAYERVALSPROPAGATOR_H_*/
