#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include <vector>

#include "diffFunc.h"
#include "../commonConsts.h"
#include "../commonTypes.h"

//#include "itkConstNeighborhoodIterator.h"
//#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "neighbourhoodIterator.h"
// tools
#include "tools/cellRemoteArray.h"
#include "tools/cellLinkedChainIterator.h"

namespace M4D {
namespace Cell {

template <typename TValuePixel, typename TFeaturePixel, uint8 Dimension >
class UpdateCalculatorSPE
	: public CommonTypes<Dimension>
	, public Consts<TValuePixel, typename CommonTypes<Dimension>::StatusType>
{
public:
	
	/** Standard class typedefs */
	typedef CommonTypes<Dimension> SuperClass;
	typedef typename SuperClass::TimeStepType TimeStepType;
	typedef typename SuperClass::StatusType StatusType;
	
	typedef TValuePixel ValueType;
	typedef TFeaturePixel FeaturePixelType;
		
	typedef NeighborhoodCell<FeaturePixelType, Dimension> TFeatureNeighbourhood;
	typedef NeighborhoodCell<ValueType, Dimension> TOutputNeighbourhood;
	
	typedef NeighbourIteratorCell<TValuePixel, Dimension> TOutIter;
	typedef NeighbourIteratorCell<TFeaturePixel, Dimension> TFeatureIter;
	
	typedef ThresholdLevelSetFunc<TOutIter, TFeatureIter> SegmentationFunctionType;
	typedef RemoteArrayCell<TValuePixel, 8> TUpdateBufferArray;
		
		  
  typedef typename SegmentationFunctionType::FloatOffsetType 	FloatOffsetType;
  typedef typename SegmentationFunctionType::NeighborhoodScalesType NeighborhoodScalesType;
  
  typedef typename TOutIter::IndexType IndexType;
  /** Node type used in sparse field layer lists. */
  typedef SparseFieldLevelSetNode<IndexType> LayerNodeType;
  
  typedef LinkedChainIteratorCell<LayerNodeType> TLayerIterator;
  
  typedef GlobalDataStruct<FeaturePixelType, Dimension> TGlobalData;
	
	UpdateCalculatorSPE()
	{
		memset(&m_globalData, 0, sizeof(TGlobalData));
	}
		  
		  void CalculateChangeItem(void);
	
		  TimeStepType CalculateChange();
		  
		  typedef RunConfiguration<
		      	  	NeighborhoodScalesType, 
		      	  FeaturePixelType, 
		      	ValueType,
		      	LayerNodeType,
		      	  Dimension> TRunConf;
		  
		  TRunConf m_Conf;
		  
		  void Init(void);
	
protected:

	SegmentationFunctionType m_diffFunc;
    
	TUpdateBufferArray m_updateBufferArray;
	TLayerIterator m_layerIterator;
    
    
	//CalculateChangeStepConfiguration<LayerNodeType> m_stepConf;
	
	//LayerListType m_Layers;	// TODO load
	  
private:
	
	TOutIter m_outIter;
	TFeatureIter m_featureIter;
	
	ValueType MIN_NORM;
	
	TGlobalData m_globalData;
	
	// tmp variables to avoid repeating allocations on stack
	typename SegmentationFunctionType::FloatOffsetType offset;
	ValueType norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
	backwardValue, centerValue;
	unsigned i;
};

	//include implementation
	#include "src/updateCalculatorSPE.tcc"
	
} }

#endif /*HARDWORKER_H_*/
