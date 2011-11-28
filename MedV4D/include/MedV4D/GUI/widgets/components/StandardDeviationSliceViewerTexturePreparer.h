/**
 *  @ingroup gui
 *  @file StandardDeviationSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef STANDARD_DEVIATION_SLICEVIEWER_TEXTURE_PREPARER_H
#define STANDARD_DEVIATION_SLICEVIEWER_TEXTURE_PREPARER_H

#include "MedV4D/GUI/widgets/components/IntensitySummarizerSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that sets the output pixel value according to the standard deviation within a given radius of the input images
 */
template< typename ElementType >
class StandardDeviationSliceViewerTexturePreparer : public virtual IntensitySummarizerSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    StandardDeviationSliceViewerTexturePreparer() : _radius( 1 ) {}

    /**
     * Set the radius of edge value calculation
     *  @param radius the new radius of the edge value calculation
     */
    void SetRadius( uint32 radius )
	{ _radius = radius; }

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
     * Calculates the standard deviation of the surrounding points to the given point of the image
     *  @param channel pointer to the array of pixels
     *  @param xcoord the x coordinate of the given point
     *  @param ycoord the y coordinate of the given point
     *  @param width the width of the image
     *  @param height the height of the image
     *  @return the standard deviation value
     */
    double StandardDeviation(
        ElementType* channel,
        uint32 xcoord,
        uint32 ycoord,
        uint32 width,
        uint32 height );

    /** The radius of the standard deviation calculation */
    int32 _radius;

private:

    StandardDeviationSliceViewerTexturePreparer( const StandardDeviationSliceViewerTexturePreparer& ); // not implemented
    const StandardDeviationSliceViewerTexturePreparer& operator=( const StandardDeviationSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/StandardDeviationSliceViewerTexturePreparer.tcc"

#endif /*STANDARD_DEVIATION_SLICEVIEWER_TEXTURE_PREPARER_H*/
