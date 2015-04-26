#ifndef RENDERER_TOOLS_H
#define RENDERER_TOOLS_H

#include <QtCore>
#include "MedV4D/Common/Common.h"
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
	ctBasic,
	ctSimpleColorMap,
  ctEigenvalues,
  ctEigenvaluesRaw,

	ctIsoSurfaces // TODO - remove
};

enum MultiDatasetRenderingStyle
{
	mdrsOnlyPrimary,
	mdrsMask
};

struct QStringNameIdPair
{
	QStringNameIdPair( QString aName, unsigned aId ): name( aName ), id( aId )
	{ }

	QString name;
	unsigned id;
};

//typedef std::vector< WideNameIdPair > ColorTransformNameIDList;
typedef std::vector< QStringNameIdPair > ColorTransformNameIDList;

}//Renderer
}//GUI
}//M4D


#endif //RENDERER_TOOLS_H
