/**
 *  @ingroup gui
 *  @file MedianGradientSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MEDIAN_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define MEDIAN_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/MedianIntensitySliceViewerTexturePreparer.h"
#include "GUI/widgets/components/GradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that selects the median of the input pixel gradients
 */
template< typename ElementType >
class MedianGradientSliceViewerTexturePreparer : public virtual MedianIntensitySliceViewerTexturePreparer< ElementType >,
						 public virtual GradientSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    MedianGradientSliceViewerTexturePreparer() {}

private:

    MedianGradientSliceViewerTexturePreparer( const MedianGradientSliceViewerTexturePreparer& ); // not implemented
    const MedianGradientSliceViewerTexturePreparer& operator=( const MedianGradientSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*MEDIAN_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H*/
