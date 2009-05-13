#ifndef CONFIGSTRUCTURES_H_
#define CONFIGSTRUCTURES_H_

#include "commonTypes.h"

namespace M4D {
namespace Cell {

enum ESPUCommands
{
	CALC_PROPAG_VALS,
	CALC_CHANGE,
	CALC_UPDATE,
	QUIT
};

///////////////////////////////////////////////////////////////////////////////

#define PropagateValuesConf_Allign 64
#define PropagateValuesConf_AllignExponent 6

struct PropagateValuesConf
{
	Address layerBegins[LYERCOUNT];
	Address layerEnds[LYERCOUNT];
	
	void operator=(const PropagateValuesConf& o)
	{
		for(uint32 i=0; i<LYERCOUNT; i++)
		{
		layerBegins[i] = o.layerBegins[i];
		layerEnds[i] = o.layerEnds[i];
		}
	}
} __attribute__((aligned(PropagateValuesConf_Allign)));

///////////////////////////////////////////////////////////////////////////////

#define CalculateChangeAndUpdActiveLayerConf_Allign 32
#define CalculateChangeAndUpdActiveLayerConf_AllignExponent 5

struct CalculateChangeAndUpdActiveLayerConf
{
	Address layer0Begin;
	Address layer0End;

	Address updateBuffBegin;
	
	void operator=(const CalculateChangeAndUpdActiveLayerConf& o)
	{
		layer0Begin = o.layer0Begin;
		layer0End = o.layer0End;
		updateBuffBegin = o.updateBuffBegin;
	}
} __attribute__((aligned(CalculateChangeAndUpdActiveLayerConf_Allign)));

///////////////////////////////////////////////////////////////////////////////

// geather all configurations that SPE needs to load
#define RunConfiguration_Allign 128
#define RunConfiguration_AllignExponent 7

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
    
    uint32 SPEId;
} __attribute__((aligned(RunConfiguration_Allign)));

///////////////////////////////////////////////////////////////////////////////

#define ConfigStructures_Allign 
#define ConfigStructures_AllignExponent 5

class ConfigStructures
{
public:
	Address runConf;
	Address calcChngApplyUpdateConf;
	Address propagateValsConf;
} __attribute__ ((aligned(32)));

///////////////////////////////////////////////////////////////////////////////

}
}  // namespace

#endif /*CONFIGSTRUCTURES_H_*/
