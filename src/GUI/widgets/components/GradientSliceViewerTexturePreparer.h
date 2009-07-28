/**
 *  @ingroup gui
 *  @file GradientSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that calculates the gradient of the input datasets
 */
template< typename ElementType >
class GradientSliceViewerTexturePreparer : public virtual SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    GradientSliceViewerTexturePreparer() {}

protected:

    /**
     * Prepares several texture arrays of datasets - consisting of the gradients of the image array
     *  @param inputPorts the input pipeline port list to get the image from
     *  @param numberOfDatasets the number of datasets to be arranged and returned
     *  @param width reference to set the width of the texture
     *  @param height reference to set the height of the texture
     *  @param so the orientation of the slices (xy, yz, zx)
     *  @param slice the number of the slice to be drawn
     *  @param dimension dataset's number of dimensions
     *  @return array of arrays of the prepared textures
     */
    ElementType** getDatasetArrays( const Imaging::InputPortList& inputPorts,
      uint32 numberOfDatasets,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension );


private:

    GradientSliceViewerTexturePreparer( const GradientSliceViewerTexturePreparer& ); // not implemented
    const GradientSliceViewerTexturePreparer& operator=( const GradientSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

//include source
#include "src/GradientSliceViewerTexturePreparer.tcc"

#endif /*GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H*/
