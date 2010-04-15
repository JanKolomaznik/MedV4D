#ifndef TYPE_DECLARATIONS_H
#define TYPE_DECLARATIONS_H


#include "common/Common.h"
#include "common/Vector.h"


#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "PerfusionStudies"

#define EXEMINED_SLICE_NUM     2

#define BONE_DENSITY_BOTTOM    1300
#define BONE_DENSITY_TOP       4000


typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;

#include "Imaging/filters/ImageConvertor.h"

#include "MultiscanRegistrationFilter.h"
#include "MultiscanSegmentationFilter.h"
#include "PerfusionAnalysisFilter.h"

// Filters
typedef M4D::Imaging::ImageConvertor< ImageType >              Convertor;
typedef M4D::Imaging::MultiscanRegistrationFilter< ImageType > Registration;
typedef M4D::Imaging::MultiscanSegmentationFilter< ImageType > Segmentation;
typedef M4D::Imaging::PerfusionAnalysisFilter< ImageType >     Analysis;


#endif // TYPE_DECLARATIONS_H
