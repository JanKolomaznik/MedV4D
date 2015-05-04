#ifndef ITK_EIGENVALUES_H_
#define ITK_EIGENVALUES_H_

#include <itkImage.h>

#include <itkHessianRecursiveGaussianImageFilter.h>

#include <MedV4D/Common/Types.h>

namespace M4D
{
namespace GUI
{
namespace Viewer
{
template<typename PixelType = unsigned short, typename EigenvalueType = double, unsigned int Dimension = 3>
class ItkEigenvalues
{
public:
	typedef itk::Image<PixelType, Dimension> ItkImageType;
	typedef typename ItkImageType::ConstPointer ItkImageConstPointer;
	typedef typename ItkImageType::Pointer ItkImagePointer;

	typedef itk::Vector<EigenvalueType, Dimension> EigenvaluesType;
	typedef itk::Image<EigenvaluesType, Dimension> EigenvaluesCollectionType;
	typedef Vector<EigenvalueType, Dimension> EigenvaluesOutputType;
	typedef itk::Image<EigenvaluesOutputType, Dimension> EigenvaluesOutputCollectionType;
	typedef itk::ImageRegionIterator<EigenvaluesOutputCollectionType> EigenvaluesOutputCollectionIteratorType;
	typedef itk::ImageRegionIterator<EigenvaluesCollectionType> EigenvaluesCollectionIteratorType;

	typedef itk::SymmetricSecondRankTensor<EigenvalueType, Dimension> TensorType;
	typedef itk::Image<TensorType, Dimension> HessianOutputType;
	typedef typename HessianOutputType::Pointer HessianOutputTypePointer;
	typedef itk::ImageRegionConstIterator<HessianOutputType>  HessianIteratorType;
	typedef itk::HessianRecursiveGaussianImageFilter<ItkImageType, HessianOutputType> HessianFilterType;

	ItkEigenvalues(ItkImagePointer image, PixelType hessianSigma)
		: image(image),
		  hessianSigma(hessianSigma),
		  eigenvaluesPerVoxel(EigenvaluesCollectionType::New()),
		  eigenvaluesPerVoxelOutput(EigenvaluesOutputCollectionType::New())
	{
	}

	typename ItkEigenvalues::EigenvaluesOutputCollectionType::Pointer GetEigenValuesImage()
	{
		this->GetEigenValues();

		std::cout << "Performing computes on eigenvalues" << std::endl;

		return this->ConvertEigenvalues();
	}

	void GetEigenValues()
	{
		// TODO: cache computed eigenvalues somewhere
		this->ComputeEigenvalues();
	}

	void ComputeEigenvalues()
	{
		double hessianSigma = this->hessianSigma;

		std::cout << "computing hessian matrices" << std::endl;

		typename HessianOutputType::Pointer hessianOutput = this->GetHessianRecursiveGaussianFilterImage(hessianSigma);

		std::cout << "computing eigenvalues at each point" << std::endl;

		this->InitializeEigenvalues<EigenvaluesCollectionType>(this->eigenvaluesPerVoxel);

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

	template<typename EigenvaluesCollection>
	void InitializeEigenvalues(typename EigenvaluesCollection::Pointer collection) const
	{
		typename EigenvaluesCollection::SizeType imageSize = this->image->GetLargestPossibleRegion().GetSize();
		typename EigenvaluesCollection::SizeType size;
		typename EigenvaluesCollection::IndexType start;
		double origin[Dimension];
		double spacing[Dimension];

		for (size_t i = 0; i < Dimension; ++i)
		{
			size[i] = imageSize[i];

			// these values does not really matter, because they are only iterated and computed values are stored into original image (that has correctly set spacing etc.)
			start[i] = 0;
			origin[i] = 0.0;
			spacing[i] = 1.0;
		}

		typename EigenvaluesCollection::RegionType region;
		region.SetSize(size);
		region.SetIndex(start);

		collection->SetRegions(region);
		collection->SetOrigin(origin);
		collection->SetSpacing(spacing);
		collection->Allocate();
		typename EigenvaluesCollection::PixelType default_value(0.0);
		collection->FillBuffer(default_value);
	}

	typename EigenvaluesOutputCollectionType::Pointer ConvertEigenvalues() const
	{
		typename EigenvaluesOutputCollectionType::Pointer eigenvaluesOutput = EigenvaluesOutputCollectionType::New();
		this->InitializeEigenvalues<EigenvaluesOutputCollectionType>(eigenvaluesOutput);

		EigenvaluesOutputCollectionIteratorType eigenValuesOutputIterator(eigenvaluesOutput, eigenvaluesOutput->GetRequestedRegion());
		EigenvaluesCollectionIteratorType eigenValuesIterator(this->eigenvaluesPerVoxel, this->eigenvaluesPerVoxel->GetRequestedRegion());
		for (eigenValuesOutputIterator.GoToBegin(); !eigenValuesOutputIterator.IsAtEnd(); ++eigenValuesOutputIterator, ++eigenValuesIterator)
		{
			EigenvaluesType eigenvalues = eigenValuesIterator.Get();
			EigenvaluesOutputType eigenvaluesOutput;

			for (size_t i = 0; i < Dimension; ++i)
			{
				eigenvaluesOutput[i] = eigenvalues[i];
			}

			eigenValuesOutputIterator.Set(eigenvaluesOutput);
		}

		return eigenvaluesOutput;
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

	ItkImagePointer image;

	PixelType hessianSigma;

	typename EigenvaluesCollectionType::Pointer eigenvaluesPerVoxel;
	typename EigenvaluesOutputCollectionType::Pointer eigenvaluesPerVoxelOutput;
};
}
}
}

#endif // ITK_EIGENVALUES_H_
