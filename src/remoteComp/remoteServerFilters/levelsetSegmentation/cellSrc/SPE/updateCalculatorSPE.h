#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include <vector>

#include "diffFunc.h"
#include "../commonConsts.h"
#include "../commonTypes.h"

#include "itkSparseFieldLayer.h"

//#include "itkConstNeighborhoodIterator.h"
//#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "neighbourhoodIterator.h"

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
		
	typedef NeighbourIteratorCell<FeaturePixelType, Dimension> TFeatureNeighbourhood;
	typedef NeighbourIteratorCell<ValueType, Dimension> TOutputNeighbourhood;		
	typedef ThresholdLevelSetFunc<TOutputNeighbourhood, TFeatureNeighbourhood> SegmentationFunctionType;
		
		  
  typedef typename SegmentationFunctionType::FloatOffsetType 	FloatOffsetType;
  typedef typename SegmentationFunctionType::NeighborhoodScalesType NeighborhoodScalesType;
  
  // spolecne !!!!!!!!!!!!!!
  typedef typename TOutputNeighbourhood::IndexType IndexType;
  /** Node type used in sparse field layer lists. */
  typedef itk::SparseFieldLevelSetNode<IndexType> LayerNodeType;
  
  /** A list type used in the algorithm. */
  typedef itk::SparseFieldLayer<LayerNodeType> LayerType;
  typedef typename LayerType::Pointer     LayerPointerType;
  typedef std::vector<LayerPointerType> LayerListType;
  
  typedef GlobalDataStruct<FeaturePixelType, Dimension> TGlobalData;
  
	
	/** Container type used to store updates to the active layer. */
	typedef std::vector<ValueType> UpdateBufferType;
  // !!!!!!!!!!!!!!!!!!!!!!!!
	
	UpdateCalculatorSPE()
	{
		memset(&m_globalData, 0, sizeof(TGlobalData));
	}
		  
		  //void CalculateChangeItem(NeighborhoodIterator<OutputImageType> &outIt);
	
		  TimeStepType CalculateChange();
		  
		  typedef RunConfiguration<
		      	  	NeighborhoodScalesType, 
		      	  FeaturePixelType, 
		      	ValueType,
		      	  	LayerType,
		      	  	UpdateBufferType,
		      	  Dimension> TRunConf;
		  
		  TRunConf m_Conf;
		  
		  void Init(void);
	
protected:

	SegmentationFunctionType m_diffFunc;
    
    
    
    
	//CalculateChangeStepConfiguration<LayerNodeType> m_stepConf;
	
	//LayerListType m_Layers;	// TODO load
	  
private:
	
	
	
	ValueType MIN_NORM;
	
	TGlobalData m_globalData;
};

	//include implementation
	#include "src/updateCalculatorSPE.tcc"
	
} }

#endif /*HARDWORKER_H_*/
