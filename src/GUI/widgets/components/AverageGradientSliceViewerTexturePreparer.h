/**
 *  @ingroup gui
 *  @file AverageGradientSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef AVERAGE_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define AVERAGE_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/AverageIntensitySliceViewerTexturePreparer.h"
#include "GUI/widgets/components/GradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that calculates the average gradient for each pixel
 */
template< typename ElementType >
class AverageGradientSliceViewerTexturePreparer : public virtual AverageIntensitySliceViewerTexturePreparer< ElementType >,
						  public virtual GradientSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    AverageGradientSliceViewerTexturePreparer() {}

private:

    AverageGradientSliceViewerTexturePreparer( const AverageGradientSliceViewerTexturePreparer& ); // not implemented
    const AverageGradientSliceViewerTexturePreparer& operator=( const AverageGradientSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*AVERAGE_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H*/
