/**
 *  @ingroup gui
 *  @file MultiChannelRGBSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MULTI_CHANNEL_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#define MULTI_CHANNEL_RGB_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/RGBSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class MultiChannelRGBSliceViewerTexturePreparer : public RGBSliceViewerTexturePreparer< ElementType >
{

public:

    MultiChannelRGBSliceViewerTexturePreparer() {}

    /**
     * Prepares the texture of the image to be mapped to the following OpenGL surface.
     *  @param inputPorts the input pipeline port list to get the image from
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param brightnessRate the rate of brightness to adjust the image with
     *  @param contrastRate the rate of contrast to adjust the image with
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dimense
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

private:

    MultiChannelRGBSliceViewerTexturePreparer( const MultiChannelRGBSliceViewerTexturePreparer& ); // not implemented
    const MultiChannelRGBSliceViewerTexturePreparer& operator=( const MultiChannelRGBSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/MultiChannelRGBSliceViewerTexturePreparer.tcc"

#endif /*MULTI_CHANNEL_RGB_SLICEVIEWER_TEXTURE_PREPARER_H*/
