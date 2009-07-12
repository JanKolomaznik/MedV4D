/**
 *  @ingroup gui
 *  @file MinimumGradientSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MINIMUM_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define MINIMUM_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/MinimumIntensitySliceViewerTexturePreparer.h"
#include "GUI/widgets/components/GradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class MinimumGradientSliceViewerTexturePreparer : public virtual MinimumIntensitySliceViewerTexturePreparer< ElementType >,
						  public virtual GradientSliceViewerTexturePreparer< ElementType >
{

public:

    MinimumGradientSliceViewerTexturePreparer() {}

private:

    MinimumGradientSliceViewerTexturePreparer( const MinimumGradientSliceViewerTexturePreparer& ); // not implemented
    const MinimumGradientSliceViewerTexturePreparer& operator=( const MinimumGradientSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*MINIMUM_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H*/
