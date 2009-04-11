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

class LayerValuesPropagator : public Consts, public LayerGate
{
public:
	void PropagateAllLayerValues();			
			
	ApplyUpdateConf conf;
	
protected:
	
	typedef LinkedChainIteratorCell<SparseFieldLevelSetNode> TLayerIterator;
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	TLayerIterator m_layerIterator;
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;
	
private:
	void PropagateLayerValues(StatusType from, StatusType to,
				                       StatusType promote, uint32 InOrOut);
};

}
}
#endif /*LAYERVALSPROPAGATOR_H_*/
