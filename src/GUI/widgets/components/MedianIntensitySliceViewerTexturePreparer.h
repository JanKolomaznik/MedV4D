/**
 *  @ingroup gui
 *  @file MedianIntensitySliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MEDIAN_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#define MEDIAN_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/IntensitySummarizerSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class MedianIntensitySliceViewerTexturePreparer : public virtual IntensitySummarizerSliceViewerTexturePreparer< ElementType >
{

public:

    MedianIntensitySliceViewerTexturePreparer() {}

protected:

    /**
     * Arranges the input arrays into one array of image that contains one of the input values of each position
     *  @param channels the values of the images in the channels
     *  @param channelNumber the number of the channels
     *  @param width the width of the image
     *  @param height the height of the image
     *  @return the prepared texture array
     */
    ElementType* IntensityArranger(
        ElementType** channels,
	uint32 channelNumber,
        uint32 width,
        uint32 height );

private:

    MedianIntensitySliceViewerTexturePreparer( const MedianIntensitySliceViewerTexturePreparer& ); // not implemented
    const MedianIntensitySliceViewerTexturePreparer& operator=( const MedianIntensitySliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/MedianIntensitySliceViewerTexturePreparer.tcc"

#endif /*MEDIAN_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H*/
