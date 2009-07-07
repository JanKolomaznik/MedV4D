#ifndef SHAREDRESOURCES_H_
#define SHAREDRESOURCES_H_

#include "../configStructures.h"

namespace M4D {
namespace Cell {

/**
 * This struct has to gather all big statically allocated objects
 * to allow some of them to be reused within both updateCalc and
 * ApplyUpdate phases
 */ 
struct SharedResources
{
	RunConfiguration _runConf __attribute__ ((aligned (128)));	
	CalculateChangeAndUpdActiveLayerConf _changeConfig __attribute__ ((aligned (128)));
	PropagateValuesConf _propValConfig __attribute__ ((aligned (128)));
	
	// buffers for remote arrays operations
	// defined here to be shareable between all (update, applyupdate) 
	// calculators
	TPixelValue _buf[2][REMOTEARRAY_BUF_SIZE] __attribute__ ((aligned (128)));
};

}
}
#endif /*SHAREDRESOURCES_H_*/
