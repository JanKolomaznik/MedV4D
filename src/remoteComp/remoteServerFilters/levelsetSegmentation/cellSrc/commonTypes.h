#ifndef COMMONTYPES_H_
#define COMMONTYPES_H_

namespace itk {

// geather all configurations that SPE needs to load
template<typename NeighborhoodScalesType, typename FeatureImageType, 
typename OutputImageType, typename TInputImage, typename LayerListType, 
typename UpdateBufferType>
class RunConfiguration
{
public:
	
	typedef RunConfiguration<NeighborhoodScalesType, FeatureImageType, 
		OutputImageType, TInputImage, LayerListType, 
		UpdateBufferType> Self;
	
	typename FeatureImageType::PixelType m_upThreshold;
	typename FeatureImageType::PixelType m_downThreshold;
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
    
	const FeatureImageType *m_featureImage;
	OutputImageType *m_outputImage;
	const TInputImage *m_inputImage;
	
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
		m_featureImage = o.m_featureImage;
		m_outputImage = o.m_outputImage;
		m_inputImage = o.m_inputImage;
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

}  // namespace itk

#endif /*COMMONTYPES_H_*/
