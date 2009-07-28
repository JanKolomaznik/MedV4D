/**
 *  @ingroup gui
 *  @file AverageIntensitySliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef AVERAGE_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#define AVERAGE_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/IntensitySummarizerSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that calculates the average value of each pixel position
 */
template< typename ElementType >
class AverageIntensitySliceViewerTexturePreparer : public virtual IntensitySummarizerSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    AverageIntensitySliceViewerTexturePreparer() {}

protected:

    /**
     * Arranges the input arrays into one array of image that contains one of the input values of each position
     *  @param channels the values of the images in the channels
     *  @param channelNumber the number of the channels
     *  @param width the width of the image
     *  @param height the height of the image
     *  @return the prepared texture array
     */
    virtual ElementType* IntensityArranger(
        ElementType** channels,
	uint32 channelNumber,
        uint32 width,
        uint32 height );

private:

    AverageIntensitySliceViewerTexturePreparer( const AverageIntensitySliceViewerTexturePreparer& ); // not implemented
    const AverageIntensitySliceViewerTexturePreparer& operator=( const AverageIntensitySliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/AverageIntensitySliceViewerTexturePreparer.tcc"

#endif /*AVERAGE_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H*/
