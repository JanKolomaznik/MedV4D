/**
 *  @ingroup gui
 *  @file RGBSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#define RGB_SLICEVIEWER_TEXTURE_PREPARER_H

#include "MedV4D/GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer that shows the first three input datasets by assigning
 * each one of them to one of the channels of RGB
 */
template< typename ElementType >
class RGBSliceViewerTexturePreparer : public virtual SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    RGBSliceViewerTexturePreparer();

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

    /**
     * Set brightness and contrast modifications
     *  @param multiply red channel's brightness
     *  @param multiply green channel's brightness
     *  @param multiply blue channel's brightness
     *  @param multiply red channel's contrast
     *  @param multiply green channel's contrast
     *  @param multiply blue channel's contrast
     */
    void setAdjustBrightnessContrast(
      double brightnessAdjustRed,
      double brightnessAdjustGreen,
      double brightnessAdjustBlue,
      double contrastAdjustRed,
      double contrastAdjustGreen,
      double contrastAdjustBlue );


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
        uint32 height,
        GLint brightnessRate,
        GLint contrastRate,
	bool adjustBrightnessContrast = true );

    /** multiply red channel's brightness */
    double _brightnessAdjustRed;

    /** multiply green channel's brightness */
    double _brightnessAdjustGreen;

    /** multiply blue channel's brightness */
    double _brightnessAdjustBlue;

    /** multiply red channel's contrast */
    double _contrastAdjustRed;

    /** multiply green channel's contrast */
    double _contrastAdjustGreen;

    /** multiply blue channel's contrast */
    double _contrastAdjustBlue;

private:

    RGBSliceViewerTexturePreparer( const RGBSliceViewerTexturePreparer& ); // not implemented
    const RGBSliceViewerTexturePreparer& operator=( const RGBSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/RGBSliceViewerTexturePreparer.tcc"

#endif /*RGB_SLICEVIEWER_TEXTURE_PREPARER_H*/
