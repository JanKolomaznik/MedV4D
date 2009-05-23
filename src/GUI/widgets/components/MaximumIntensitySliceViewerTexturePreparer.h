/**
 *  @ingroup gui
 *  @file MaximumIntensitySliceViewerTexturePreparer.h
 *  @brief some brief
 */
#ifndef MAXIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#define MAXIMUM_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H

#include "GUI/widgets/components/IntensitySelectorSliceViewerTexturePreparer.h"


namespace M4D
{
namespace Viewer
{

template< typename ElementType >
class MaximumIntensityComparator : public IntensityComparator< ElementType >
{

public:

    bool operator()( const ElementType& operand1, const ElementType& operand2 )
    {
	return operand1 < operand2;
    }

};

template< typename ElementType >
class MaximumIntensitySliceViewerTexturePreparer : public IntensitySelectorSliceViewerTexturePreparer< ElementType >
{

public:

    MaximumIntensitySliceViewerTexturePreparer() : IntensitySelectorSliceViewerTexturePreparer< ElementType >( new MaximumIntensityComparator< ElementType >() ) {}

    ~MaximumIntensitySliceViewerTexturePreparer()
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
