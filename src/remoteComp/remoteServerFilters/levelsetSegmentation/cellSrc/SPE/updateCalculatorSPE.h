#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include "diffFunc.h"
//#include "../commonConsts.h"
#include "configStructures.h"

//#include "itkConstNeighborhoodIterator.h"
//#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "neighbourhoodIterator.h"
// tools
#include "tools/cellRemoteArray.h"
#include "tools/cellLinkedChainIterator.h"

namespace M4D {
namespace Cell {

class UpdateCalculatorSPE
//	: public CommonTypes<Dimension>
//	, public Consts<TValuePixel, typename CommonTypes<Dimension>::StatusType>
{
public:
	
	/** Standard class typedefs */
//	typedef CommonTypes<Dimension> SuperClass;
//	typedef typename SuperClass::TimeStepType TimeStepType;
//	typedef typename SuperClass::StatusType StatusType;
//	
//	typedef TValuePixel ValueType;
//	typedef TFeaturePixel FeaturePixelType;
//		
//	typedef NeighborhoodCell<FeaturePixelType, Dimension> TFeatureNeighbourhood;
//	typedef NeighborhoodCell<ValueType, Dimension> TOutputNeighbourhood;
//	
//	typedef NeighbourIteratorCell<TValuePixel, Dimension> TOutIter;
//	typedef NeighbourIteratorCell<TFeaturePixel, Dimension> TFeatureIter;
//	
//	typedef ThresholdLevelSetFunc<TOutIter, TFeatureIter> SegmentationFunctionType;
	typedef RemoteArrayCell<TPixelValue, 8> TUpdateBufferArray;
//		
//		  
//  typedef typename SegmentationFunctionType::FloatOffsetType 	FloatOffsetType;
//  typedef typename SegmentationFunctionType::NeighborhoodScalesType NeighborhoodScalesType;
//  
//  typedef typename TOutIter::IndexType IndexType;
//  /** Node type used in sparse field layer lists. */
  typedef SparseFieldLevelSetNode LayerNodeType;
  
  typedef LinkedChainIteratorCell<LayerNodeType> TLayerIterator;
//  
//  typedef GlobalDataStruct<FeaturePixelType, Dimension> TGlobalData;
	
  UpdateCalculatorSPE();
		  
  void CalculateChangeItem(void);

  TimeStepType CalculateChange();
  
  
  RunConfiguration m_Conf;
		  
		  void Init(void);
	
protected:

	ThresholdLevelSetFunc m_diffFunc;
    
	TUpdateBufferArray m_updateBufferArray;
	TLayerIterator m_layerIterator;
    
    
	//CalculateChangeStepConfiguration m_stepConf;
	
	//LayerListType m_Layers;	// TODO load
	  
private:
	
	NeighbourIteratorCell m_outIter;
	NeighbourIteratorCell m_featureIter;
	
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
