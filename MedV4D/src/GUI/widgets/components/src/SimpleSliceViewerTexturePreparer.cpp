#include "MedV4D/GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"

namespace M4D
{
namespace Viewer
{

template<>
GLenum
SimpleSliceViewerTexturePreparer<uint8>::oglType()
{
    return GL_UNSIGNED_BYTE;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<int8>::oglType()
{
    return GL_BYTE;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<uint16>::oglType()
{
    return GL_UNSIGNED_SHORT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<int16>::oglType()
{
    return GL_SHORT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<uint32>::oglType()
{
    return GL_UNSIGNED_INT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<int32>::oglType()
{
    return GL_INT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<float32>::oglType()
{
    return GL_FLOAT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<uint64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_UNSIGNED_INT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<int64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_INT;
}

template<>
GLenum
SimpleSliceViewerTexturePreparer<float64>::oglType()
{
    throw ErrorHandling::ExceptionBase( "64-bit numbers are not supported." );
    return GL_FLOAT;
}

} /*namespace Viewer*/
} /*namespace M4D*/
