#pragma once

#include <itkImage.h>
#include "MedV4D/Common/Vector.h"

#include "MedV4D/Imaging/AImage.h"
#include "MedV4D/Imaging/Image.h"
#include <boost/filesystem.hpp>
#include <prognot/prognot.hpp>

M4D::Imaging::AImage::Ptr
loadItkImage(const boost::filesystem::path &aFile, prognot::ProgressNotifier aProgressNotifier);


template<typename TElement>
typename M4D::Imaging::Image<TElement, 3>::Ptr
itkImageToM4dImage(typename itk::Image<TElement, 3>::Pointer aImage);


template<typename TElement>
typename itk::Image<TElement, 3>::Pointer
M4dImageToItkImage(typename M4D::Imaging::Image<TElement, 3>::ConstPtr aImage)
{
  const int cDimension = 3;
  typedef itk::Image<TElement, cDimension> ItkImageType;

  Vector<float, cDimension> minimum = aImage->GetMinimum();
  Vector<float, cDimension> maximum = aImage->GetMaximum();
  typename M4D::Imaging::Image<TElement, 3>::ElementExtentsType elementExtents = aImage->GetElementExtents();

  ItkImageType::SizeType size;
  ItkImageType::IndexType start;
  double origin[cDimension];
  double spacing[cDimension];

  for (size_t i = 0; i < 3; ++i)
  {
    size[i] = maximum[i] - minimum[i];
    start[i] = 0;
    origin[i] = 0.0;
    spacing[i] = elementExtents[i];
  }

  ItkImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  typename ItkImageType::Pointer itkImage = ItkImageType::New();
  itkImage->SetRegions(region);
  itkImage->Allocate();
  itkImage->SetOrigin(origin);
  itkImage->SetSpacing(spacing);


  ItkImageType::IndexType index = region.GetIndex();
  size_t width = size[0];
  size_t height = size[1];
  size_t depth = size[2];

  for (int k = minimum[2]; k < maximum[2]; ++k) {
    for (int j = minimum[1]; j < maximum[1]; ++j) {
      for (int i = minimum[0]; i < maximum[0]; ++i) {
        Vector<int32, cDimension> coords(i, j, k);
        typename ItkImageType::IndexType index;
        index[0] = i;
        index[1] = j;
        index[2] = k;

        TElement value = aImage->GetElement(coords);
        itkImage->SetPixel(index, value);
      }
    }
  }

  return itkImage;
}