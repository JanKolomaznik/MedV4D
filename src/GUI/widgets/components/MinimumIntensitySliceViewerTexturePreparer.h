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

template< typename ElementType >
class MinimumIntensityComparator : public IntensityComparator< ElementType >
{

public:

    bool operator()( const ElementType& operand1, const ElementType& operand2 )
    {
	return operand1 > operand2;
    }

};

template< typename ElementType >
class MinimumIntensitySliceViewerTexturePreparer : public IntensitySelectorSliceViewerTexturePreparer< ElementType >
{

public:

    MinimumIntensitySliceViewerTexturePreparer() : IntensitySelectorSliceViewerTexturePreparer< ElementType >( new MinimumIntensityComparator< ElementType >() ) {}

    ~MinimumIntensitySliceViewerTexturePreparer()
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
