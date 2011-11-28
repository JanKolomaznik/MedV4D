/**
 *  @ingroup gui
 *  @file MaximumGradientSliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MAXIMUM_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#define MAXIMUM_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H

#include "MedV4D/GUI/widgets/components/MaximumIntensitySliceViewerTexturePreparer.h"
#include "MedV4D/GUI/widgets/components/GradientSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Sliceviewer's texture preparer component that selects the maximum gradient value of the input dataset pixels
 */
template< typename ElementType >
class MaximumGradientSliceViewerTexturePreparer : public virtual MaximumIntensitySliceViewerTexturePreparer< ElementType >,
						  public virtual GradientSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    MaximumGradientSliceViewerTexturePreparer() {}

private:

    MaximumGradientSliceViewerTexturePreparer( const MaximumGradientSliceViewerTexturePreparer& ); // not implemented
    const MaximumGradientSliceViewerTexturePreparer& operator=( const MaximumGradientSliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*MAXIMUM_GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H*/
