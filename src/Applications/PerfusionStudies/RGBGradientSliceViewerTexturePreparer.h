/**
 * @author Attila Ulman
 * @file RGBGradientSliceViewerTexturePreparer.h
 * @{ 
 **/

#ifndef RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"

#include "GUI/widgets/ogl/fonts.h"

#include <sstream>


namespace M4D {
namespace Viewer {

#define	MAX_VALUE             255

#define	RAMP_POS_X            0.9
#define	RAMP_POS_Y            0.5
#define	RAMP_WIDTH            20
#define	RAMP_HEIGHT           200

#define	FONT_HEIGHT           16
#define	FONT_WIDTH            8
#define	FONT_SPACING          2

#define	BRIGHTNESS_FACTOR     12
#define	CONTRAST_FACTOR       20

/**
 * Sliceviewer's texture preparer that shows the first three input datasets by assigning
 * each one of them to one of the channels of RGB
 */
template< typename ElementType >
class RGBGradientSliceViewerTexturePreparer 
  : public virtual SimpleSliceViewerTexturePreparer< ElementType >
{
  public:

    /**
     * Constructor
     */
    RGBGradientSliceViewerTexturePreparer ();

    void setMinMaxValue ( uint16 min, uint16 max );

    /**
     * Prepares the texture of the image to be mapped to the following OpenGL surface.
     *
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
    virtual bool prepare ( const Imaging::InputPortList &inputPorts,
                           uint32 &width, uint32 &height,
                           GLint brightnessRate, GLint contrastRate,
                           SliceOrientation so,
                           uint32 slice,
                           unsigned &dimension );

  private:

    void ColorRamp ( double x, ElementType *rgb, uint32 idx, uint32 channelSize );

    void DrawRamp ( ElementType *rgb, uint32 width, uint32 height );

    void DrawValue ( uint16 value, uint32 xPos, uint32 yPos, ElementType *rgb, uint32 width, uint32 height );

    RGBGradientSliceViewerTexturePreparer ( const RGBGradientSliceViewerTexturePreparer &preparer );

    const RGBGradientSliceViewerTexturePreparer &operator = ( const RGBGradientSliceViewerTexturePreparer &preparer );

    uint16 minValue, maxValue;
};

} // namespace Viewer
} // namespace M4D


//include implementation
#include "RGBGradientSliceViewerTexturePreparer.tcc"

#endif // RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

/** @} */
