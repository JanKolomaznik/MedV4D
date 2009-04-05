#ifndef COMMONTYPES_H_
#define COMMONTYPES_H_

#include "SPE/neighborhoodCell.h"

namespace M4D {
namespace Cell {

// geather all configurations that SPE needs to load
template<typename NeighborhoodScalesType, typename FeaturePixelType, 
typename ValuePixelType, typename LayerListType, 
typename UpdateBufferType, uint8 Dimension>
class RunConfiguration
{
public:
	
	typedef RunConfiguration<NeighborhoodScalesType, FeaturePixelType, 
		ValuePixelType, LayerListType, UpdateBufferType, Dimension> Self;
	
	typedef typename NeighborhoodCell<ValuePixelType, Dimension>::TImageProperties ValueImageProps;
	typedef typename NeighborhoodCell<FeaturePixelType, Dimension>::TImageProperties FeatureImageProps;
	
	FeaturePixelType m_upThreshold;
	ValuePixelType m_downThreshold;
    float32 m_propWeight;
    float32 m_curvWeight;
    
	  /** The number of layers to use in the sparse field.  Sparse field will
	   * consist of m_NumberOfLayers layers on both sides of a single active layer.
	   * This active layer is the interface of interest, i.e. the zero
	   * level set. */
	uint8 m_NumberOfLayers;
    
    /** The constant gradient to maintain between isosurfaces in the
    	      sparse-field of the level-set image.  This value defaults to 1.0 */
    float64 m_ConstantGradientValue;
  
    NeighborhoodScalesType m_neighbourScales;
    
    UpdateBufferType *m_UpdateBuffer;
	LayerListType *m_activeSet;
    
	FeatureImageProps featureImageProps;
	ValueImageProps valueImageProps;
	
	void operator=(const Self& o)
	{
		m_upThreshold = o.m_upThreshold;
		m_downThreshold = o.m_downThreshold;
		m_propWeight = o.m_propWeight;
		m_curvWeight = o.m_curvWeight;
		m_NumberOfLayers = o.m_NumberOfLayers;
		m_ConstantGradientValue = o.m_ConstantGradientValue;
		m_neighbourScales = o.m_neighbourScales;
		m_UpdateBuffer = o.m_UpdateBuffer;
		m_activeSet = o.m_activeSet;
		featureImageProps = o.featureImageProps;
		valueImageProps = o.valueImageProps;
	}
};

//template<typename TNode>
//class CalculateChangeStepConfiguration
//{
//public:
//	static const uint32 itemCountInOneRun = 10 * 1024;
//	
//	TNode *begin;
//	TNode *end;
//	
//	TimeStepType dt;
//};

}
}  // namespace

#endif /*COMMONTYPES_H_*/
