#ifndef SPEHARDWORKER_H_
#define SPEHARDWORKER_H_

#include <vector>

#include "diffFunc.h"
#include "../commonConsts.h"
#include "../commonTypes.h"

#include "itkSparseFieldLayer.h"

#include "itkConstNeighborhoodIterator.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"

namespace itk {

template <class TInputImage, class TFeatureImage, class TOutputPixelType = float >
class UpdateCalculatorSPE
	: public CommonTypes<TInputImage::ImageDimension>
	, public Consts<typename Image<TOutputPixelType, TInputImage::ImageDimension>::ValueType, typename CommonTypes<TInputImage::ImageDimension>::StatusType>
{
public:
	
	/** Standard class typedefs */
	typedef CommonTypes<TInputImage::ImageDimension> SuperClass;
	typedef typename SuperClass::TimeStepType TimeStepType;
	typedef typename SuperClass::StatusType StatusType;
	
		typedef Image<TOutputPixelType, TInputImage::ImageDimension> OutputImageType;
		typedef TInputImage                         InputImageType;
		typedef TFeatureImage                       FeatureImageType;
		typedef typename FeatureImageType::PixelType FeaturePixelType;
		
		// &&&&&&&&&
		typedef ZeroFluxNeumannBoundaryCondition<TInputImage>  DefaultBoundaryConditionType;
		typedef ConstNeighborhoodIterator<TInputImage, DefaultBoundaryConditionType> InNeighborhoodType;
		typedef ConstNeighborhoodIterator<TFeatureImage, DefaultBoundaryConditionType> FeatureNeighborhoodType;
		// &&&&&&&&&
		
		typedef ThresholdLevelSetFunc<InNeighborhoodType, FeatureNeighborhoodType> SegmentationFunctionType;
		
		  typedef typename Image<TOutputPixelType, TInputImage::ImageDimension>::ValueType ValueType;
		  
		  typedef typename SegmentationFunctionType::FloatOffsetType 	FloatOffsetType;

		  /**Typedefs from the superclass */
	//	  typedef typename SegmentationFunctionType::RadiusType             RadiusType;
		  typedef typename SegmentationFunctionType::NeighborhoodScalesType NeighborhoodScalesType;
		  
		  // spolecne !!!!!!!!!!!!!!
		  typedef typename OutputImageType::IndexType IndexType;
		  /** Node type used in sparse field layer lists. */
		  typedef SparseFieldLevelSetNode<IndexType> LayerNodeType;
		  
		  /** A list type used in the algorithm. */
		  typedef SparseFieldLayer<LayerNodeType> LayerType;		  
		  typedef typename LayerType::Pointer     LayerPointerType;		  
		  typedef std::vector<LayerPointerType> LayerListType;
		  
		  typedef Image<StatusType, TInputImage::ImageDimension> StatusImageType;
		  typedef GlobalDataStruct<FeaturePixelType, FeatureImageType::ImageDimension> TGlobalData;
		  
			
			/** Container type used to store updates to the active layer. */
			typedef std::vector<ValueType> UpdateBufferType;
		  // !!!!!!!!!!!!!!!!!!!!!!!!
			
			UpdateCalculatorSPE()
			{
				memset(&m_globalData, 0, sizeof(TGlobalData));
			}
		  
		  void CalculateChangeItem(NeighborhoodIterator<OutputImageType> &outIt);
	
		  TimeStepType CalculateChange();
		  
		  typedef RunConfiguration<
		      	  	NeighborhoodScalesType, 
		      	  	FeatureImageType, 
		      	  	OutputImageType, 
		      	  	InputImageType,
		      	  	LayerType,
		      	  	UpdateBufferType> TRunConf;
		  
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

}

//include implementation
#include "src/updateCalculatorSPE.tcc"

#endif /*HARDWORKER_H_*/
