#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include "diffFunc.h"
//#include "../commonConsts.h"
#include "../configStructures.h"
#include "../tools/neighbourhoodIterator.h"
// tools
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"
#include "../tools/preloadedNeighbourhoods.h"

namespace M4D {
namespace Cell {

class UpdateCalculatorSPE
{
public:
	
	typedef TPixelValue TFeature;

	typedef RemoteArrayCell<TPixelValue, 8> TUpdateBufferArray;
	
	typedef NeighborhoodCell<TPixelValue> TValueNeighborhood;
	typedef NeighborhoodCell<TFeature> TFeatureNeighborhood;
	
	typedef PreloadedNeigborhoods<TValueNeighborhood, 4> TValueNeighbPreloadeder;
	typedef PreloadedNeigborhoods<TFeatureNeighborhood, 4> TFeatureNeighbPreloadeder;
	
  typedef SparseFieldLevelSetNode LayerNodeType;
  
  typedef LinkedChainIteratorCell<LayerNodeType> TLayerIterator;
	
  UpdateCalculatorSPE();
		  
  void CalculateChangeItem(void);

  TimeStepType CalculateChange();  
  
  RunConfiguration *m_Conf;
  CalculateChangeAndUpdActiveLayerConf *m_stepConfig;
		  
		  void Init(void);
		  void UpdateFunctionProperties();
	
protected:

	ThresholdLevelSetFunc m_diffFunc;
    
	TUpdateBufferArray m_updateBufferArray;
	TLayerIterator m_layerIterator;
	TValueNeighbPreloadeder m_valueNeighbPreloader;
	TValueNeighbPreloadeder m_featureNeighbPreloader;
	  
private:
	
	typedef NeighbourIteratorCell<TPixelValue> TNeighbourIterator;
	
	TNeighbourIterator m_outIter;
	TNeighbourIterator m_featureIter;
	
	TPixelValue MIN_NORM;
	
	GlobalDataStruct m_globalData;
	
	// tmp variables to avoid repeating allocations on stack
	TContinuousIndex offset;
	TPixelValue norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
	backwardValue, centerValue;
	unsigned i;
};

//	//include implementation
//	#include "src/updateCalculatorSPE.tcc"
	
} }

#endif /*HARDWORKER_H_*/
