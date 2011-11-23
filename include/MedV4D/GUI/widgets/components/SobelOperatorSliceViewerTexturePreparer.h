/**
 *  @ingroup gui
 *  @file SobelOperatorSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef SOBEL_OPERATOR_SLICEVIEWER_TEXTURE_PREPARER_H
#define SOBEL_OPERATOR_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/IntensitySummarizerSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that sets the output pixel value according to the output of the Sobel operator applied to the pixels of the input images
 */
template< typename ElementType >
class SobelOperatorSliceViewerTexturePreparer : public virtual IntensitySummarizerSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    SobelOperatorSliceViewerTexturePreparer() {}

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

    /**
     * Applies the Sobel operator to a given point of the image
     *  @param channel pointer to the array of pixels
     *  @param xcoord the x coordinate of the given point
     *  @param ycoord the y coordinate of the given point
     *  @param width the width of the image
     *  @param height the height of the image
     *  @return the Sobel operator value
     */
    ElementType SobelOperator(
	ElementType* channel,
	uint32 xcoord,
	uint32 ycoord,
	uint32 width,
	uint32 height );

private:

    SobelOperatorSliceViewerTexturePreparer( const SobelOperatorSliceViewerTexturePreparer& ); // not implemented
    const SobelOperatorSliceViewerTexturePreparer& operator=( const SobelOperatorSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/SobelOperatorSliceViewerTexturePreparer.tcc"

#endif /*SOBEL_OPERATOR_SLICEVIEWER_TEXTURE_PREPARER_H*/
