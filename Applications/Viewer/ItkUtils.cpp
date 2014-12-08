#include "ItkUtils.hpp"


#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include <boost/filesystem.hpp>

#include "MedV4D/Imaging/ImageFactory.h"
#include "MedV4D/Imaging/AImage.h"
#include "MedV4D/Imaging/Image.h"

#include <prognot/prognot.hpp>
#include "MedV4D/Common/ExceptionBase.h"

#include <map>

template<typename TElement>
typename M4D::Imaging::Image<TElement, 3>::Ptr
itkImageToM4dImage(typename itk::Image<TElement,3>::Pointer aImage)
{
	const int cDimension = 3;
	typedef itk::Image<TElement, cDimension> ItkImageType;

	Vector<int32, cDimension> minimum;
	Vector<int32, cDimension> maximum;
	Vector<float32, cDimension> elementExtents;
	auto region = aImage->GetLargestPossibleRegion();

	for (int i = 0; i < cDimension; ++i) {
		minimum[i] = region.GetIndex()[i];
		maximum[i] = minimum[i] + region.GetSize()[i] ;
		elementExtents[i] = aImage->GetSpacing()[i];
	}

	auto image = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents
			<TElement, 3>(minimum, maximum, elementExtents);

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
readItkImage(const boost::filesystem::path &aFile, prognot::ProgressNotifier aProgressNotifier)
{
	typedef itk::Image<TElement, 3> ImageType;
	typedef itk::ImageFileReader<ImageType> ReaderType;

	typename ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(aFile.string());
	reader->Update();
	return reader->GetOutput();
}

template<typename TElement>
M4D::Imaging::AImage::Ptr
loadItkImageTyped(const boost::filesystem::path &aFile, prognot::ProgressNotifier aProgressNotifier)
{
	aProgressNotifier.setStepCount(3);
	auto image = readItkImage<TElement>(aFile, aProgressNotifier.subTaskNotifier(2));
	return itkImageToM4dImage<TElement>(image);
}

M4D::Imaging::AImage::Ptr
loadItkImage(const boost::filesystem::path &aFile, prognot::ProgressNotifier aProgressNotifier)
{
	typedef M4D::Imaging::AImage::Ptr(*Loader)(const boost::filesystem::path &, prognot::ProgressNotifier);
	const std::map<itk::ImageIOBase::IOComponentType, Loader> kLoadFunctions = {
			{ itk::ImageIOBase::UCHAR, &loadItkImageTyped<unsigned char> },
			{ itk::ImageIOBase::CHAR, &loadItkImageTyped<signed char> },
			{ itk::ImageIOBase::USHORT, &loadItkImageTyped<unsigned short> },
			{ itk::ImageIOBase::SHORT, &loadItkImageTyped<short> },
			{ itk::ImageIOBase::UINT, &loadItkImageTyped<unsigned int> },
			{ itk::ImageIOBase::INT, &loadItkImageTyped<int> },
			{ itk::ImageIOBase::ULONG, &loadItkImageTyped<unsigned long> },
			{ itk::ImageIOBase::LONG, &loadItkImageTyped<long> },
			{ itk::ImageIOBase::FLOAT, &loadItkImageTyped<float> },
			{ itk::ImageIOBase::DOUBLE, &loadItkImageTyped<double> }
		};

	aProgressNotifier.setStepCount(4);
	typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

	itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(
				aFile.string().c_str(),
				itk::ImageIOFactory::ReadMode);
	imageIO->SetFileName(aFile.string().c_str());
	imageIO->ReadImageInformation();
	const ScalarPixelType pixelType = imageIO->GetComponentType();

	if (imageIO->GetNumberOfComponents () != 1) {
		M4D_THROW(M4D::ErrorHandling::io_error()); //TODO - better exception
	}
	auto it = kLoadFunctions.find(pixelType);
	if (it == kLoadFunctions.end()) {
		M4D_THROW(M4D::ErrorHandling::io_error()); //TODO - better exception
	}
	aProgressNotifier.increment();
	return (*(it->second))(aFile, aProgressNotifier.subTaskNotifier(3));
}

