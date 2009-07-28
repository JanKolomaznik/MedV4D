/**
 *  @ingroup gui
 *  @file MinimumIntensitySliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MINIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#define MINIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/IntensitySelectorSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

/**
 * Comparator class that returns true if the second operator is smaller than the first
 */
template< typename ElementType >
class MinimumIntensityComparator : public IntensityComparator< ElementType >
{

public:

    /**
     * The comparing operator itself
     */
    bool operator()( const ElementType& operand1, const ElementType& operand2 )
    {
	return operand1 > operand2;
    }

};

/**
 * Sliceviewer's texture preparer component that selects the minimum intensity of the input pixel values
 */
template< typename ElementType >
class MinimumIntensitySliceViewerTexturePreparer : public IntensitySelectorSliceViewerTexturePreparer< ElementType >
{

public:

    /**
     * Constructor
     */
    MinimumIntensitySliceViewerTexturePreparer() : IntensitySelectorSliceViewerTexturePreparer< ElementType >( new MinimumIntensityComparator< ElementType >() ) {}

    /**
     * Destructor
     */
    virtual ~MinimumIntensitySliceViewerTexturePreparer()
    {
	delete this->_comparator;
    }

private:

    MinimumIntensitySliceViewerTexturePreparer( const MinimumIntensitySliceViewerTexturePreparer& ); // not implemented
    const MinimumIntensitySliceViewerTexturePreparer& operator=( const MinimumIntensitySliceViewerTexturePreparer& ); // not implemented

};

} /*namespace Viewer*/
} /*namespace M4D*/

#endif /*MINIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H*/
