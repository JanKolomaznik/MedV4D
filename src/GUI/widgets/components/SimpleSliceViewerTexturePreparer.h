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

/**
 * Sliceviewer's texture preparer that shows the first input dataset as a greyscale image
 */
template< typename ElementType >
class SimpleSliceViewerTexturePreparer : public AbstractSliceViewerTexturePreparer
{

public:

    /**
     * Constructor
     */
    SimpleSliceViewerTexturePreparer() {}

    /**
     * Get the OpenGL enum constant for a given type - different implementation
     * for each template specialization.
     *  @return OpenGL enum constant for the given type
     */
    GLenum oglType();

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

protected:

    /**
     * Function that arranges the voxels in correct order.
     *  @param dst pointer to the destination array
     *  @param src pointer to the source array
     *  @param width the width of the image
     *  @param height the height of the image
     *  @param newWidth the new width of the image after texture correction ( to be a power of 2 )
     *  @param newHeight the new height of the image after texture correction ( to be a power of 2 )
     *  @param depth the depth at which the slice lies
     *  @param xstride the steps between two neighbor voxels according to coordinate x
     *  @param ystride the steps between two neighbor voxels according to coordinate y
     *  @param zstride the steps between two neighbor voxels according to coordinate z
     */
    void copy( ElementType* dst, ElementType* src, uint32 width, uint32 height, uint32 newWidth, uint32 newHeight, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
    {
        uint32 i, j;
        for ( i = 0; i < newHeight; i++ )
            for ( j = 0; j < newWidth; j++ )
		if ( i < height && j < width ) dst[ i * newWidth + j ] = src[ j * xstride + i * ystride + depth * zstride ];
		else dst[ i * newWidth + j ] = 0;
    }

    /**
     * Prepares the texture of the image to be mapped to the following OpenGL surface.
     *  @param inPort the input port to get the image from
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return pointer to the resulting texture array, if texture preparing was successful, NULL otherwise
     */
    ElementType* prepareSingle( Imaging::InputPortTyped<Imaging::AbstractImage>* inPort,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );

    /**
     * Prepares several texture arrays of datasets
     *  @param inputPorts the input pipeline port list to get the image from
     *  @param numberOfDatasets the number of datasets to be arranged and returned
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return array of arrays of the prepared textures
     */
    virtual ElementType** getDatasetArrays( const Imaging::InputPortList& inputPorts,
      uint32 numberOfDatasets,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );

    /** Equlizes the histogram of the image array to the given brightness and contrast rates
     *  @param width the width of the image array
     *  @param height the height of the image array
     *  @param brightnessRate the rate of brightness to adjust the image with
     *  @param contrastRate the rate of contrast to adjust the image with
     */
    void adjustArrayContrastBrightness( ElementType* pixel,
      uint32 width,
      uint32 height,
      GLint brightnessRate,
      GLint contrastRate );


private:

    SimpleSliceViewerTexturePreparer( const SimpleSliceViewerTexturePreparer& ); // not implemented
    const SimpleSliceViewerTexturePreparer& operator=( const SimpleSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/SimpleSliceViewerTexturePreparer.tcc"

#endif /*SIMPLE_SLICEVIEWER_TEXTURE_PREPARER_H*/
