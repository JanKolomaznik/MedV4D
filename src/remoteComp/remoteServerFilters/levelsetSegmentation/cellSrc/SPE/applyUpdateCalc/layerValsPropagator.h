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



namespace M4D {
namespace Cell {

class LayerValuesPropagator : public Consts
{
public:
	
	void PropagateAllLayerValues();
			
	PropagateValuesConf *m_propLayerValuesConfig;
	RunConfiguration *commonConf;
	
	LayerGate m_layerGate;
	
protected:
	
	LayerValuesPropagator();
	
	typedef LinkedChainIteratorCell<SparseFieldLevelSetNode> TLayerIterator;
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	typedef NeighborhoodCell<TPixelValue> TValueNeighborhood;
	typedef NeighborhoodCell<StatusType> TStatusNeighborhood;
	typedef PreloadedNeigborhoods<TValueNeighborhood, 4> TValueNeighbPreloadeder;
	typedef PreloadedNeigborhoods<TStatusNeighborhood, 4> TStatusNeighbPreloadeder;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;	
	
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