/**
 * @author Attila Ulman
 * @file RGBGradientSliceViewerTexturePreparer.h
 * @{ 
 **/

#ifndef RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "MedV4D/GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"

#include "MedV4D/GUI/widgets/ogl/fonts.h"

#include <sstream>


namespace M4D {
namespace Viewer {

#define	MAX_VALUE             255

#define	SEE_THROUGH_RADIUS    50

#define	RAMP_POS_X            0.9
#define	RAMP_POS_Y            0.5
#define	RAMP_WIDTH            20
#define	RAMP_HEIGHT           200

#define	FONT_HEIGHT           16
#define	FONT_WIDTH            8
#define	FONT_SPACING          2

#define	BRIGHTNESS_FACTOR     12
#define	CONTRAST_FACTOR       20

#define	BGR_BRIGHTNESS_FACTOR 8
#define	BGR_CONTRAST_FACTOR   4

/**
 * Sliceviewer's texture preparer that allows displaying parameter maps with see-through 
 * interface support.
 */
template< typename ElementType >
class RGBGradientSliceViewerTexturePreparer 
  : public virtual SimpleSliceViewerTexturePreparer< ElementType >
{
  public:

    /**
     * Texture preparer constructor.
     */
    RGBGradientSliceViewerTexturePreparer ();

    /**
     * Sets min. and max. values occuring in the parameter map.
     *
     *  @param min the min. value in the parameter map
     *  @param max the max. value in the parameter map
     */
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

    /**
     * Maps scalar value to color map (hot-to-cold).
     *
     *  @param x the value to be mapped (0 - 1.0)
     *  @param rgb pointer to output channels
     *  @param idx index of the value within the channel
     *  @param channelSize size of one channel
     */
    void ColorRamp ( double x, ElementType *rgb, uint32 idx, uint32 channelSize );

    /**
     * Draws the see-through (circle shaped - at top of the parameter map in the texture).
     *
     *  @param rgb pointer to output channels
     *  @param background the background which can be seen through the cut
     *  @param width width of the texture
     *  @param height height of the texture
     */
    void DrawCut ( ElementType *rgb, ElementType *background, uint32 width, uint32 height );

    /**
     * Draws the color ramp to the corner of the texture.
     *
     *  @param rgb pointer to output channels
     *  @param width width of the texture
     *  @param height height of the texture
     */
    void DrawRamp ( ElementType *rgb, uint32 width, uint32 height );

    /**
     * Draws the given text to the texture.
     *
     *  @param value the value to be drawn to the texture
     *  @param xPos the x position of the text
     *  @param yPos the y position of the text
     *  @param rgb pointer to output channels
     *  @param width width of the texture
     *  @param height height of the texture
     */
    void DrawValue ( uint16 value, uint32 xPos, uint32 yPos, ElementType *rgb, uint32 width, uint32 height );

    /// Min. and max. values occuring in the parameter map (for color ramp calibration).
    uint16 minValue, maxValue;
};

} // namespace Viewer
} // namespace M4D


//include implementation
#include "RGBGradientSliceViewerTexturePreparer.tcc"

#endif // RGB_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

/** @} */
