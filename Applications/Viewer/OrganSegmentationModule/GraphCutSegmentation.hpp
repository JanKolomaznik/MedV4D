#pragma once

#include <MedV4D/Imaging/Image.h>

void
computeGraphCutSegmentation(const M4D::Imaging::AImageDim<3> &aImage, const M4D::Imaging::Mask3D &aMarkerData, M4D::Imaging::Mask3D &aSegmentationMask);

