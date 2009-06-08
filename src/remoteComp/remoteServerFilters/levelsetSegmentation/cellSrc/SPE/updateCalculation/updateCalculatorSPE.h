#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include "diffFunc.h"
#include "../tools/neighbourhoodIterator.h"
// tools
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"
#include "../tools/preloadedNeighbourhoods.h"

#include "../tools/sharedResources.h"

namespace M4D
{
namespace Cell
{

class UpdateCalculatorSPE
{
public:

	typedef TPixelValue TFeature;

	typedef RemoteArrayCell<TPixelValue, REMOTEARRAY_BUF_SIZE> TUpdateBufferArray;

	typedef NeighborhoodCell<TPixelValue> TValueNeighborhood;
	typedef NeighborhoodCell<TFeature> TFeatureNeighborhood;

	typedef PreloadedNeigborhoods<TPixelValue, 4> TValueNeighbPreloadeder;
	typedef PreloadedNeigborhoods<TFeature, 4>	TFeatureNeighbPreloadeder;

	typedef SparseFieldLevelSetNode LayerNodeType;

	typedef LinkedChainIteratorCell<LayerNodeType> TLayerIterator;
	typedef NeighbourIteratorCell<TPixelValue> TNeighbourIterator;

	UpdateCalculatorSPE(SharedResources *shaRes);

	void CalculateChangeItem(void);
	TimeStepType CalculateChange();



	void Init(void);
	void UpdateFunctionProperties();

protected:
	ThresholdLevelSetFunc m_diffFunc;
	TUpdateBufferArray m_updateBufferArray;
	TLayerIterator m_layerIterator;
	TValueNeighbPreloadeder m_valueNeighbPreloader;
	TValueNeighbPreloadeder m_featureNeighbPreloader;

private:
	TNeighbourIterator m_outIter;
	TNeighbourIterator m_featureIter;
	TPixelValue MIN_NORM;
	GlobalDataStruct m_globalData;
	
	RunConfiguration *m_Conf;
	CalculateChangeAndUpdActiveLayerConf *m_stepConfig;

	// tmp variables to avoid repeating allocations on stack
	TContinuousIndex offset;
	TPixelValue norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
			backwardValue, centerValue;
	unsigned i;
};

}
}

#endif /*HARDWORKER_H_*/
