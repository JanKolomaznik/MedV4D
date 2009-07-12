/**
 *  @ingroup gui
 *  @file MaxMinMinGradientRGBSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MAX_MIN_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#define MAX_MIN_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/RGBSliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MaximumIntensitySliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MinimumIntensitySliceViewerTexturePreparer.h"
#include "GUI/widgets/components/MinimumGradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class MaxMinMinGradientRGBSliceViewerTexturePreparer :	public RGBSliceViewerTexturePreparer< ElementType >,
						public virtual MaximumIntensitySliceViewerTexturePreparer< ElementType >,
						public virtual MinimumIntensitySliceViewerTexturePreparer< ElementType >,
						public virtual MinimumGradientSliceViewerTexturePreparer< ElementType >
{

public:

    MaxMinMinGradientRGBSliceViewerTexturePreparer() {}

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

    /**
     * Arranges the input arrays into one array of image that contains one of the input values of each position
     * NOT usable in this class, only in predecessors!
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
        uint32 height )
        {
            return 0;
        }

    MaxMinMinGradientRGBSliceViewerTexturePreparer( const MaxMinMinGradientRGBSliceViewerTexturePreparer& ); // not implemented
    const MaxMinMinGradientRGBSliceViewerTexturePreparer& operator=( const MaxMinMinGradientRGBSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/MaxMinMinGradientRGBSliceViewerTexturePreparer.tcc"

#endif /*MAX_MIN_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H*/
