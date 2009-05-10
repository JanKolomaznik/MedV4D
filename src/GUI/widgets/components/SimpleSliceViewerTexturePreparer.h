/**
 *  @ingroup gui
 *  @file SimpleSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef SIMPLE_SLICEVIEWER_TEXTURE_PREPARER_H
#define SIMPLE_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/AbstractSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class SimpleSliceViewerTexturePreparer : public AbstractSliceViewerTexturePreparer< ElementType >
{

public:

    SimpleSliceViewerTexturePreparer() {}

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
    bool prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );

private:

    /**
     * Function that arranges the voxels in correct order.
     *  @param dst pointer to the destination array
     *  @param src pointer to the source array
     *  @param width the width of the image
     *  @param height the height of the image
     *  @param newWidth the new width of the image after texture correction ( to be a power of 2 )
     *  @param depth the depth at which the slice lies
     *  @param xstride the steps between two neighbor voxels according to coordinate x
     *  @param ystride the steps between two neighbor voxels according to coordinate y
     *  @param zstride the steps between two neighbor voxels according to coordinate z
     */
    void copy( ElementType* dst, ElementType* src, uint32 width, uint32 height, uint32 newWidth, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
    {
        uint32 i, j;
        for ( i = 0; i < height; i++ )
            for ( j = 0; j < width; j++ ) dst[ i * newWidth + j ] = src[ j * xstride + i * ystride + depth * zstride ];
    }

    SimpleSliceViewerTexturePreparer( const SimpleSliceViewerTexturePreparer& ); // not implemented
    const SimpleSliceViewerTexturePreparer& operator=( const SimpleSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/SimpleSliceViewerTexturePreparer.tcc"

#endif /*SIMPLE_SLICEVIEWER_TEXTURE_PREPARER_H*/
