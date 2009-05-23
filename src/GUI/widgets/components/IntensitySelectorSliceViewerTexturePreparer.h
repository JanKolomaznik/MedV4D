/**
 *  @ingroup gui
 *  @file IntensitySelectorSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef INTENSITY_SELECTOR_SLICEVIEWER_TEXTURE_PREPARER_H
#define INTENSITY_SELECTOR_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"


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
class IntensitySelectorSliceViewerTexturePreparer : public SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    IntensitySelectorSliceViewerTexturePreparer( IntensityComparator< ElementType >* comp ) : _comparator( comp ) {}

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
     * Arranges the input arrays into one array of image that contains one of the input values of each position
     *  @param channels the values of the images in the channels
     *  @param channelNumber the number of the channels
     *  @param width the width of the image
     *  @param height the height of the image
     *  @return the prepared texture array
     */
    ElementType* IntensitySelectorArranger(
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
