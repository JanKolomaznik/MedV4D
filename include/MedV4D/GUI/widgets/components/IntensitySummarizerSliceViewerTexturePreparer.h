/**
 *  @ingroup gui
 *  @file IntensitySummarizerSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef INTENSITY_SUMMARIZER_SLICEVIEWER_TEXTURE_PREPARER_H
#define INTENSITY_SUMMARIZER_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's component - abstract base for pixel summarizer classes
 */
template< typename ElementType >
class IntensitySummarizerSliceViewerTexturePreparer : public virtual SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    IntensitySummarizerSliceViewerTexturePreparer() {}

    /**
     * Prepares the texture of the image to be mapped to the following OpenGL surface.
     *  @param inputPorts the input pipeline port list to get the image from
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param brightnessRate the rate of brightness to adjust the image with
     *  @param contrastRate the rate of contrast to adjust the image with
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return true, if texture preparing was successful, false otherwise
     */
    virtual bool prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );

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
        uint32 height ) = 0;

private:

    IntensitySummarizerSliceViewerTexturePreparer( const IntensitySummarizerSliceViewerTexturePreparer& ); // not implemented
    const IntensitySummarizerSliceViewerTexturePreparer& operator=( const IntensitySummarizerSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/IntensitySummarizerSliceViewerTexturePreparer.tcc"

#endif /*INTENSITY_SUMMARIZER_SLICEVIEWER_TEXTURE_PREPARER_H*/
