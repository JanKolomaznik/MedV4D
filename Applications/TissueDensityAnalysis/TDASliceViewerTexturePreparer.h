/**
 * @ingroup TDA 
 * @author Milan Lepik
 * @file TDASliceViewerWidget.h 
 * @{ 
 **/

#ifndef _TDA_SLICE_VIEWER_TEXTURE_PREPARER_H
#define _TDA_SLICE_VIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/SimpleSliceViewerTexturePreparer.h"

using namespace M4D;
using namespace M4D::Viewer;

/**
 * Sliceviewer's texture preparer that shows the first input dataset as a greyscale image
 */
template< typename ElementType >
class TDASliceViewerTexturePreparer : public SimpleSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    TDASliceViewerTexturePreparer(): SimpleSliceViewerTexturePreparer< ElementType >() {}

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
    virtual bool prepare( 
		const Imaging::InputPortList& inputPorts,
		uint32& width,
		uint32& height,
		GLint brightnessRate,
		GLint contrastRate,
		SliceOrientation so,
		uint32 slice,
		unsigned& dimension 
	);

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
    void copy( 
		ElementType* dst, 
		ElementType* src, 
		uint32 width, 
		uint32 height, 
		uint32 newWidth, 
		uint32 newHeight, 
		uint32 depth, 
		int32 xstride, 
		int32 ystride, 
		int32 zstride 
	);

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
    void maskCopy( 
		ElementType* dst, 
		ElementType* src, 
		ElementType* mask, 
		uint32 width, 
		uint32 height, 
		uint32 newWidth, 
		uint32 newHeight, 
		uint32 depth, 
		int32 xstride, 
		int32 ystride, 
		int32 zstride 
	);
	

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
    ElementType* prepareSingle( 
		Imaging::InputPortTyped<Imaging::AImage>* inPort,
		Imaging::InputPortTyped<Imaging::AImage>* inMaskPort,
		uint32& width,
		uint32& height,
		SliceOrientation so,
		uint32 slice,
		unsigned& dimension 
	);

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
    virtual ElementType** getDatasetArrays( 
		const Imaging::InputPortList& inputPorts,
		uint32 numberOfDatasets,
		uint32& width,
		uint32& height,
		SliceOrientation so,
		uint32 slice,
		unsigned& dimension 
	);
};

#endif

/** @} */