#ifndef ITK_FILTERING_H_
#define ITK_FILTERING_H_

#include <itkImage.h>

#include <itkGradientMagnitudeImageFilter.h>
#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkHessian3DToVesselnessMeasureImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkAbsImageFilter.h>
#include <itkLaplacianRecursiveGaussianImageFilter.h>

namespace M4D
{
  namespace GUI
  {
    namespace Viewer
    {
      template<typename MethodPolicy, typename PixelType = unsigned short, typename EigenvalueType = typename MethodPolicy::InputValueType, unsigned int Dimension = 3>
      class ItkFiltering
      {
      public:
        typedef itk::Image<PixelType, Dimension> ItkImageType;
        typedef typename ItkImageType::ConstPointer ItkImageConstPointer;
        typedef typename ItkImageType::Pointer ItkImagePointer;

        typedef itk::Vector<EigenvalueType, Dimension> EigenvaluesType;
        typedef itk::Image<EigenvaluesType, Dimension> EigenvaluesCollectionType;
        typedef itk::ImageRegionIterator<EigenvaluesCollectionType> EigenvaluesCollectionIteratorType;

        typedef itk::SymmetricSecondRankTensor<EigenvalueType, Dimension> TensorType;
        typedef itk::Image<TensorType, Dimension> HessianOutputType;
        typedef typename HessianOutputType::Pointer HessianOutputTypePointer;
        typedef itk::ImageRegionConstIterator<HessianOutputType>  HessianIteratorType;
        typedef itk::HessianRecursiveGaussianImageFilter<ItkImageType, HessianOutputType> HessianFilterType;
        typedef itk::Hessian3DToVesselnessMeasureImageFilter<PixelType> VesselnessMeasureFilterType;
        typedef itk::Image<double, Dimension> FloatImageType;
        typedef itk::RescaleIntensityImageFilter<ItkImageType, ItkImageType> RescaleFilterType;
        typedef itk::RescaleIntensityImageFilter<FloatImageType, ItkImageType> RescaleFilterIntegerType;
        typedef itk::LaplacianRecursiveGaussianImageFilter<ItkImageType, FloatImageType>  FilterType;
        typedef itk::AbsImageFilter <FloatImageType, FloatImageType> AbsImageFilterType;

        const PixelType NORMALIZATION_CONSTANT = 1000;

        ItkFiltering(ItkImagePointer image, MethodPolicy policy) : image(image), policy(policy)
        {
        }

        typename ItkFiltering::ItkImagePointer GetEigenValuesFilterImage()
        {
          this->GetEigenValues();

          std::cout << "Performing computes on eigenvalues" << std::endl;

          itk::ImageRegionIterator<ItkImageType> imageIterator(this->image, this->image->GetRequestedRegion());
          EigenvaluesCollectionIteratorType eigenvaluesIterator(this->eigenvaluesPerVoxel, this->eigenvaluesPerVoxel->GetRequestedRegion());
          eigenvaluesIterator.GoToBegin();
          for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++eigenvaluesIterator, ++imageIterator)
          {
            EigenvaluesType eigenvalues = eigenvaluesIterator.Get();
            PixelType computedValue = this->policy.ComputePixelValue(eigenvalues);
            imageIterator.Set(computedValue);
          }

          return this->image;
        }

        typename ItkFiltering::ItkImagePointer GetSatoVesselnessFilterImage(float alpha1, float alpha2)
        {
          HessianOutputType::ConstPointer hessianOutput = this->GetHessianRecursiveGaussianFilterImage(this->policy.GetHessianSigma());

          VesselnessMeasureFilterType::Pointer vesselnessFilter = VesselnessMeasureFilterType::New();

          vesselnessFilter->SetInput(hessianOutput);
          vesselnessFilter->SetAlpha1(alpha1);
          vesselnessFilter->SetAlpha2(alpha2);


          // remap values
          RescaleFilterType::Pointer rescale = RescaleFilterType::New();
          rescale->SetInput(vesselnessFilter->GetOutput());
          rescale->SetOutputMinimum(0);
          rescale->SetOutputMaximum(NORMALIZATION_CONSTANT);
          rescale->Update();

          return rescale->GetOutput();
        }

        typename ItkFiltering::ItkImagePointer GetLaplacianOfGaussianFilterImage()
        {
          FilterType::Pointer laplacian = FilterType::New();
          laplacian->SetNormalizeAcrossScale(false);
          laplacian->SetInput(this->image);

          laplacian->SetSigma(this->policy.GetHessianSigma());

          AbsImageFilterType::Pointer absFilter = AbsImageFilterType::New();
          absFilter->SetInput(laplacian->GetOutput());

          // remap values
          RescaleFilterIntegerType::Pointer rescale = RescaleFilterIntegerType::New();
          rescale->SetInput(absFilter->GetOutput());
          rescale->SetOutputMinimum(0);
          rescale->SetOutputMaximum(NORMALIZATION_CONSTANT);
          rescale->Update();

          return rescale->GetOutput();
        }

      private:

        void GetEigenValues()
        {
          // TODO: cache computed eigenvalues somewhere
          this->ComputeEigenvalues();

          /*if (boost::filesystem::exists(boost::filesystem::path(this->secondaryFilename)))
          {
          std::cout << "found image containing eigenvalues" << std::endl;

          this->LoadEigenvaluesFromImage();
          }
          else
          {
          std::cout << "image containing eigenvalues not found, they will be computed" << std::endl;

          this->ComputeEigenvalues();

          this->serializeEigenvaluesAtTheEnd = true;
          }*/
        }

        void InitializeEigenvalues()
        {
          typename EigenvaluesCollectionType::SizeType imageSize = this->image->GetLargestPossibleRegion().GetSize();
          typename EigenvaluesCollectionType::SizeType size;
          typename EigenvaluesCollectionType::IndexType start;
          double origin[Dimension];
          double spacing[Dimension];

          for (size_t i = 0; i < Dimension; ++i)
          {
            size[i] = imageSize[i];

            // these values does not really matter, because they are only iterated and computed values are stored into original image (that has correctly set spacing etc.)
            start[i] = 0;
            origin[0] = 0.0;
            spacing[0] = 1.0;
          }

          typename EigenvaluesCollectionType::RegionType region;
          region.SetSize(size);
          region.SetIndex(start);

          this->eigenvaluesPerVoxel = EigenvaluesCollectionType::New();
          this->eigenvaluesPerVoxel->SetRegions(region);
          this->eigenvaluesPerVoxel->Allocate();
          this->eigenvaluesPerVoxel->FillBuffer(EigenvaluesType());
          this->eigenvaluesPerVoxel->SetOrigin(origin);
          this->eigenvaluesPerVoxel->SetSpacing(spacing);
        }

        HessianOutputTypePointer GetHessianRecursiveGaussianFilterImage(double sigma)
        {
          typename HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
          hessianFilter->SetInput(this->image.GetPointer());
          hessianFilter->SetSigma(sigma);
          hessianFilter->Update();

          typename HessianOutputType::Pointer output = hessianFilter->GetOutput();
          return output;
        }

        void ComputeEigenvalues()
        {
          double hessianSigma = policy.GetHessianSigma();

          std::cout << "computing hessian matrices" << std::endl;

          typename HessianOutputType::Pointer hessianOutput = this->GetHessianRecursiveGaussianFilterImage(hessianSigma);

          std::cout << "computing eigenvalues at each point" << std::endl;

          this->InitializeEigenvalues();

          HessianIteratorType it(hessianOutput, hessianOutput->GetRequestedRegion());
          EigenvaluesCollectionIteratorType eigenValuesIterator(this->eigenvaluesPerVoxel, this->eigenvaluesPerVoxel->GetRequestedRegion());
          for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++eigenValuesIterator)
          {
            EigenvaluesType eigenvalues;
            TensorType hessianMatrix = it.Get();
            hessianMatrix.ComputeEigenValues(eigenvalues);
            eigenValuesIterator.Set(eigenvalues);
          }
        }

        ItkImagePointer image;

        typename EigenvaluesCollectionType::Pointer eigenvaluesPerVoxel;

        MethodPolicy policy;
      };
    }
  }
}

#endif // ITK_FILTERING_H_
