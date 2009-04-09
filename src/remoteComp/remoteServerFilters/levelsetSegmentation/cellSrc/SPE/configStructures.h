#ifndef CONFIGSTRUCTURES_H_
#define CONFIGSTRUCTURES_H_

#include "commonTypes.h"

namespace M4D {
namespace Cell {

enum ESPUCommands
{
	CALC_CHANGE,
	CALC_UPDATE,
	QUIT
};

struct ApplyUpdateConf
{
	SparseFieldLevelSetNode **layerBegins;
	SparseFieldLevelSetNode **layerEnds;
	
	void operator=(const ApplyUpdateConf& o)
	{
		layerBegins = o.layerBegins;
		layerEnds = o.layerEnds;
	}
};

// geather all configurations that SPE needs to load
class RunConfiguration
{
public:
	
	TPixelValue m_upThreshold;
	TPixelValue m_downThreshold;
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
  
    TNeighborhoodScales m_neighbourScales;
    
    TPixelValue *m_UpdateBufferData;
    SparseFieldLevelSetNode *m_activeSetBegin;
    SparseFieldLevelSetNode *m_activeSetEnd;
    
    TImageProperties<TPixelValue> featureImageProps;
    TImageProperties<TPixelValue> valueImageProps;
    TImageProperties<StatusType> statusImageProps;
	
	void operator=(const RunConfiguration& o)
	{
		m_upThreshold = o.m_upThreshold;
		m_downThreshold = o.m_downThreshold;
		m_propWeight = o.m_propWeight;
		m_curvWeight = o.m_curvWeight;
		m_NumberOfLayers = o.m_NumberOfLayers;
		m_ConstantGradientValue = o.m_ConstantGradientValue;
		m_neighbourScales = o.m_neighbourScales;
		m_UpdateBufferData = o.m_UpdateBufferData;
		m_activeSetBegin = o.m_activeSetBegin;
		m_activeSetEnd = o.m_activeSetEnd;
		featureImageProps = o.featureImageProps;
		valueImageProps = o.valueImageProps;
		statusImageProps = o.statusImageProps;
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

#endif /*CONFIGSTRUCTURES_H_*/
