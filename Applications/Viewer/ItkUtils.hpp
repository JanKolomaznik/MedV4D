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
itkImageToM4dImage(typename itk::Image<TElement, 3>::Pointer aImage)
{
  const int cDimension = 3;
  typedef itk::Image<TElement, cDimension> ItkImageType;

  Vector<int32, cDimension> minimum;
  Vector<int32, cDimension> maximum;
  Vector<float32, cDimension> elementExtents;
  auto region = aImage->GetLargestPossibleRegion();

  for (int i = 0; i < cDimension; ++i) {
    minimum[i] = region.GetIndex()[i];
    maximum[i] = minimum[i] + region.GetSize()[i];
    elementExtents[i] = aImage->GetSpacing()[i];
  }

  auto image = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents<TElement, 3>(minimum, maximum, elementExtents);

  for (int k = minimum[2]; k < maximum[2]; ++k) {
    for (int j = minimum[1]; j < maximum[1]; ++j) {
      for (int i = minimum[0]; i < maximum[0]; ++i) {
        Vector<int32, cDimension> coords(i, j, k);
        typename ItkImageType::IndexType index;
        index[0] = i;
        index[1] = j;
        index[2] = k;
        image->GetElement(coords) = aImage->GetPixel(index);
      }
    }
  }
  return image;
}


template<typename TElement>
typename itk::Image<TElement, 3>::Pointer
M4dImageToItkImage(typename M4D::Imaging::Image<TElement, 3>::ConstPtr aImage)
{
  const int cDimension = 3;
  typedef itk::Image<TElement, cDimension> ItkImageType;

  Vector<float, cDimension> minimum = aImage->GetMinimum();
  Vector<float, cDimension> maximum = aImage->GetMaximum();
  typename M4D::Imaging::Image<TElement, 3>::ElementExtentsType elementExtents = aImage->GetElementExtents();

  typename ItkImageType::SizeType size;
  typename ItkImageType::IndexType start;
  double origin[cDimension];
  double spacing[cDimension];

  for (size_t i = 0; i < 3; ++i)
  {
    size[i] = maximum[i] - minimum[i];
    start[i] = 0;
    origin[i] = 0.0;
    spacing[i] = elementExtents[i];
  }

  typename ItkImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  typename ItkImageType::Pointer itkImage = ItkImageType::New();
  itkImage->SetRegions(region);
  itkImage->Allocate();
  itkImage->SetOrigin(origin);
  itkImage->SetSpacing(spacing);


  typename ItkImageType::IndexType index = region.GetIndex();
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
