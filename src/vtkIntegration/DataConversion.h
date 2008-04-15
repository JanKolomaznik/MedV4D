#ifndef _DATA_CONVERSION_H
#define _DATA_CONVERSION_H

#include "ImageDataTemplate.h"

#include "vtkImageData.h"
#include "ExceptionBase.h"

namespace M4D
{

namespace vtkIntegration
{

class EImpossibleVTKConversion: public ErrorHandling::ExceptionBase
{
	//TODO implement
};

template< typename ElementType >
void
SetVTKImageDataScalarType( vtkImageData & dataset )
{
	throw EImpossibleVTKConversion();	
}

template<>
void
SetVTKImageDataScalarType<float>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToFloat();
}

template<>
void
SetVTKImageDataScalarType<double>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToDouble();
}

template<>
void
SetVTKImageDataScalarType<int>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToInt();
}

template<>
void
SetVTKImageDataScalarType<unsigned int>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToUnsignedInt();
}

template<>
void
SetVTKImageDataScalarType<long>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToLong();
}

template<>
void
SetVTKImageDataScalarType<unsigned long>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToUnsignedLong();
}

template<>
void
SetVTKImageDataScalarType<short>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToShort();
}

template<>
void
SetVTKImageDataScalarType<unsigned short>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToUnsignedShort();
}

template<>
void
SetVTKImageDataScalarType<signed char>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToChar();
}

template<>
void
SetVTKImageDataScalarType<unsigned char>( vtkImageData & dataset )
{
	dataset.SetScalarTypeToUnsignedChar();
}

/**
 * Creates new instance of vtkImageData containing copy of image passed as argument.
 * @param image Instance of image, which should be converted to VTK representation.
 * @exception EImpossibleVTKConversion
 * @return New instance of vtkImageData.
 **/
template< typename ElementType >
vtkImageData*
CreateVTKImageDataFromImageData( 
		const Images::ImageDataTemplate< ElementType >& image )
{
	size_t width, height, depth;
	vtkIdType IncX, IncY, IncZ;
	vtkImageData* imageData = vtkImageData::New();

	width	= image.GetDimensionInfo( 0 ).size;
	height	= image.GetDimensionInfo( 1 ).size;
	depth	= image.GetDimensionInfo( 2 ).size;	

	//imageData->SetSpacing(voxelsize.x, voxelsize.y, voxelsize.z);
	imageData->SetDimensions(width, height, depth);
	//TODO Exception handling
	SetVTKImageDataScalarType< ElementType >( *imageData );

	imageData->GetIncrements(IncX, IncY, IncZ);

	ElementType* iPtr = (ElementType*)imageData->GetScalarPointer();

	for(size_t idxZ = 0; idxZ < depth; ++idxZ)
	{
		for(size_t idxY = 0; idxY < height; ++idxY)
		{
			for(size_t idxX = 0; idxX < width; ++idxX)
			{
				*iPtr = image.Get( idxX, idxY, idxZ );
				++iPtr;
			}
			iPtr += IncY;
		}
		iPtr += IncZ;
	}
	//std::cout << IncX << ";  " << IncY << ";  " << IncZ << ";  " << std::endl;
	return imageData;
}


}/*namespace vtkIntegration*/
}/*namespace M4D*/

#endif /*_DATA_CONVERSION_H*/
