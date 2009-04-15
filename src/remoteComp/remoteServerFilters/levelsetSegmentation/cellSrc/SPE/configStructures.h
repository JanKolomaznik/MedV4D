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

struct PropagateValuesConf
{
	SparseFieldLevelSetNode *layerBegins[LYERCOUNT];
	SparseFieldLevelSetNode *layerEnds[LYERCOUNT];
	
	void operator=(const PropagateValuesConf& o)
	{
		for(uint32 i=0; i<LYERCOUNT; i++)
		{
		layerBegins[i] = o.layerBegins[i];
		layerEnds[i] = o.layerEnds[i];
		}
	}
};

struct CalculateChangeAndUpdActiveLayerConf
{
	SparseFieldLevelSetNode *layer0Begin;
	SparseFieldLevelSetNode *layer0End;

	TPixelValue *updateBuffBegin;
	
	void operator=(const CalculateChangeAndUpdActiveLayerConf& o)
	{
		layer0Begin = o.layer0Begin;
		layer0End = o.layer0End;
		updateBuffBegin = o.updateBuffBegin;
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
    
    /** The constant gradient to maintain between isosurfaces in the
    	      sparse-field of the level-set image.  This value defaults to 1.0 */
    float64 m_ConstantGradientValue;
  
    TNeighborhoodScales m_neighbourScales;
    
    TImageProperties<TPixelValue> featureImageProps;
    TImageProperties<TPixelValue> valueImageProps;
    TImageProperties<StatusType> statusImageProps;
	
	void operator=(const RunConfiguration& o)
	{
		m_upThreshold = o.m_upThreshold;
		m_downThreshold = o.m_downThreshold;
		m_propWeight = o.m_propWeight;
		m_curvWeight = o.m_curvWeight;
		m_ConstantGradientValue = o.m_ConstantGradientValue;
		m_neighbourScales = o.m_neighbourScales;
		featureImageProps = o.featureImageProps;
		valueImageProps = o.valueImageProps;
		statusImageProps = o.statusImageProps;
	}
};

class ConfigStructures
{
public:
	RunConfiguration runConf;
	CalculateChangeAndUpdActiveLayerConf calcChngApplyUpdateConf;
	PropagateValuesConf propagateValsConf;
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
