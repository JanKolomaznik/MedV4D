/**
 *  @ingroup gui
 *  @file MaxAvrMinGradientRGBSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MAX_AVR_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#define MAX_AVR_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H

#include "MedV4D/GUI/widgets/components/RGBSliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MaximumIntensitySliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/MinimumIntensitySliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/AverageGradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that selects the maximum intensity, average gradient and
 * minimum intensity of the pixels at a given position, and sets them as the channels of RGB
 */
template< typename ElementType >
class MaxAvrMinGradientRGBSliceViewerTexturePreparer :	public RGBSliceViewerTexturePreparer< ElementType >,
						public MaximumIntensitySliceViewerTexturePreparer< ElementType >,
						public MinimumIntensitySliceViewerTexturePreparer< ElementType >,
						public AverageGradientSliceViewerTexturePreparer< ElementType >
{

public:

    /** 
     * Constructor
     */
    MaxAvrMinGradientRGBSliceViewerTexturePreparer() {}

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

    MaxAvrMinGradientRGBSliceViewerTexturePreparer( const MaxAvrMinGradientRGBSliceViewerTexturePreparer& ); // not implemented
    const MaxAvrMinGradientRGBSliceViewerTexturePreparer& operator=( const MaxAvrMinGradientRGBSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/MaxAvrMinGradientRGBSliceViewerTexturePreparer.tcc"

#endif /*MAX_AVR_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H*/
