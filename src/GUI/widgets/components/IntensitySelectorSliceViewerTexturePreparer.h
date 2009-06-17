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

template< typename ElementType >
class IntensityComparator
{

public:

    virtual bool operator()( const ElementType& operand1, const ElementType& operand2 )=0;

};

template< typename ElementType >
class IntensitySelectorSliceViewerTexturePreparer : public IntensitySummarizerSliceViewerTexturePreparer< ElementType >
{

public:

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
    ElementType* IntensityArranger(
        ElementType** channels,
	uint32 channelNumber,
        uint32 width,
        uint32 height );

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
