/**
 *  @ingroup gui
 *  @file MaximumIntensitySliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MAXIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#define MAXIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H

#include "MedV4D/GUI/widgets/components/IntensitySelectorSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Comparator component that returns true if the second element is higher than the first
 */
template< typename ElementType >
class MaximumIntensityComparator : public IntensityComparator< ElementType >
{

public:

    /**
     * The comparing operator
     */
    bool operator()( const ElementType& operand1, const ElementType& operand2 )
    {
	return operand1 < operand2;
    }

};

/**
 * Sliceviewer's texture preparer component that selects the maximum intensity value of the input pixels
 */
template< typename ElementType >
class MaximumIntensitySliceViewerTexturePreparer : public IntensitySelectorSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    MaximumIntensitySliceViewerTexturePreparer() : IntensitySelectorSliceViewerTexturePreparer< ElementType >( new MaximumIntensityComparator< ElementType >() ) {}

    /**
     * Destructor
     */
    virtual ~MaximumIntensitySliceViewerTexturePreparer()
    {
	delete this->_comparator;
    }

private:

    MaximumIntensitySliceViewerTexturePreparer( const MaximumIntensitySliceViewerTexturePreparer& ); // not implemented
    const MaximumIntensitySliceViewerTexturePreparer& operator=( const MaximumIntensitySliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*MAXIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H*/
