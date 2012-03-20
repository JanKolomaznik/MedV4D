#ifndef SIMPLE_FILTERS_H
#define SIMPLE_FILTERS_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/ImageRegion.h"

template< typename TRegionType >
void
thresholding3D( TRegionType input,  M4D::Imaging::MaskRegion3D output, typename TRegionType::ElementType aThreshold, typename TRegionType::ElementType aBelowThreshold );

#endif //SIMPLE_FILTERS_H