/**
 *  @ingroup gui
 *  @file MultiChannelGradientRGBSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MULTI_CHANNEL_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#define MULTI_CHANNEL_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/MultiChannelRGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/GradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer that gives a color to each of the input datasets and
 * summarizes them according to the pixel gradients at each position
 */
template< typename ElementType >
class MultiChannelGradientRGBSliceViewerTexturePreparer : public MultiChannelRGBSliceViewerTexturePreparer< ElementType >, GradientSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    MultiChannelGradientRGBSliceViewerTexturePreparer() {}

private:

    MultiChannelGradientRGBSliceViewerTexturePreparer( const MultiChannelGradientRGBSliceViewerTexturePreparer& ); // not implemented
    const MultiChannelGradientRGBSliceViewerTexturePreparer& operator=( const MultiChannelGradientRGBSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*MULTI_CHANNEL_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H*/
