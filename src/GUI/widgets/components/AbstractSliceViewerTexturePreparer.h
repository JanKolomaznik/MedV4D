/**
 *  @ingroup gui
 *  @file AbstractSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef ABSTRACT_SLICEVIEWER_TEXTURE_PREPARER_H
#define ABSTRACT_SLICEVIEWER_TEXTURE_PREPARER_H

#include <QtOpenGL>
#include <Imaging/Imaging.h>

#define DISPLAY_PIXEL_VALUE( PIXEL, MEAN, BRIGHTNESS, CONTRAST )\
                                        ( CONTRAST * ( PIXEL- MEAN + BRIGHTNESS ) + MEAN )

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class AbstractSliceViewerTexturePreparer
{

public:

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
      unsigned& dimension ) = 0;
};

template<>
GLenum
AbstractSliceViewerTexturePreparer<uint8>::oglType()
{
    return GL_UNSIGNED_BYTE;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<int8>::oglType()
{
    return GL_BYTE;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<uint16>::oglType()
{
    return GL_UNSIGNED_SHORT;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<int16>::oglType()
{
    return GL_SHORT;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<uint32>::oglType()
{
    return GL_UNSIGNED_INT;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<int32>::oglType()
{
    return GL_INT;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<uint64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_UNSIGNED_INT;
}

template<>
GLenum
AbstractSliceViewerTexturePreparer<int64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_INT;
}

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*ABSTRACT_SLICEVIEWER_TEXTURE_PREPARER_H*/
