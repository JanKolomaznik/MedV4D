#ifndef RENDERER_TOOLS_H
#define RENDERER_TOOLS_H

#include "common/Common.h"
#include <vector>

namespace M4D
{
namespace GUI
{
namespace Renderer
{

enum ColorTransform
{
	ctLUTWindow,
	ctTransferFunction1D,
	ctMaxIntensityProjection,
	ctSimpleColorMap
};

//typedef std::vector< WideNameIdPair > ColorTransformNameIDList;
typedef std::vector< NameIdPair > ColorTransformNameIDList;

}//Renderer
}//GUI
}//M4D


#endif //RENDERER_TOOLS_H
