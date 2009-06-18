/**
 *  @ingroup gui
 *  @file RGBSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#define RGB_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class RGBSliceViewerTexturePreparer : public virtual SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    RGBSliceViewerTexturePreparer() {}

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

protected:

    /**
     * Arranges the input arrays into one array of RGB colored image.
     *  @param channelR the values of the red channel
     *  @param channelG the values of the blue channel
     *  @param channelB the values of the green channel
     *  @param width the width of the image
     *  @param height the height of the image
     *  @return the prepared RGB texture array
     */
    ElementType* RGBChannelArranger(
        ElementType* channelR,
        ElementType* channelG,
        ElementType* channelB,
        uint32 width,
        uint32 height );

private:

    RGBSliceViewerTexturePreparer( const RGBSliceViewerTexturePreparer& ); // not implemented
    const RGBSliceViewerTexturePreparer& operator=( const RGBSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/RGBSliceViewerTexturePreparer.tcc"

#endif /*RGB_SLICEVIEWER_TEXTURE_PREPARER_H*/
