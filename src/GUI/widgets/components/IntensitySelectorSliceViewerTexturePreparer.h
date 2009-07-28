/**
 *  @ingroup gui
 *  @file IntensitySelectorSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef INTENSITY_SELECTOR_SLICEVIEWER_TEXTURE_PREPARER_H
#define INTENSITY_SELECTOR_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/IntensitySummarizerSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Intensity comparator component that decides which intensity is to be selected
 */
template< typename ElementType >
class IntensityComparator
{

public:

    /**
     * The comparing method itself
     *  @return true if the second value is to be selected
     */
    virtual bool operator()( const ElementType& operand1, const ElementType& operand2 )=0;

};

/**
 * Sliceviewer's texture preparer component that selects the pixel value out of the input datasets
 * at a given position according to the output of the comparator
 */
template< typename ElementType >
class IntensitySelectorSliceViewerTexturePreparer : public virtual IntensitySummarizerSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     *  @arg comp the comparator to select the pixel value
     */
    IntensitySelectorSliceViewerTexturePreparer( IntensityComparator< ElementType >* comp ) : _comparator( comp ) {}

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
     * The comparator to select the pixel value
     */
    IntensityComparator< ElementType >*			_comparator;

private:

    IntensitySelectorSliceViewerTexturePreparer( const IntensitySelectorSliceViewerTexturePreparer& ); // not implemented
    const IntensitySelectorSliceViewerTexturePreparer& operator=( const IntensitySelectorSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/IntensitySelectorSliceViewerTexturePreparer.tcc"

#endif /*INTENSITY_SELECTOR_SLICEVIEWER_TEXTURE_PREPARER_H*/
