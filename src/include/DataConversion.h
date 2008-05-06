#ifndef _DATA_CONVERSION_H
#define _DATA_CONVERSION_H

#include "ImageDataTemplate.h"

#include "vtkImageData.h"
#include "ExceptionBase.h"

#include "M4DCommon.h"

namespace M4D
{

namespace vtkIntegration
{



class EImpossibleVTKConversion: public ErrorHandling::ExceptionBase
{
	//TODO implement
};

int
ConvertNumericTypeIDToVTKScalarType( int NumericTypeID );

template< typename ElementType >
int
GetVTKScalarTypeIdentification()
{
	throw EImpossibleVTKConversion();
}

template<>
int
GetVTKScalarTypeIdentification<float>();

template<>
int
GetVTKScalarTypeIdentification<double>();

template<>
int
GetVTKScalarTypeIdentification<int>();

template<>
int
GetVTKScalarTypeIdentification<unsigned int>();

template<>
int
GetVTKScalarTypeIdentification<long>();

template<>
int
GetVTKScalarTypeIdentification<unsigned long>();

template<>
int
GetVTKScalarTypeIdentification<short>();
template<>
int
GetVTKScalarTypeIdentification<unsigned short>();
template<>
int
GetVTKScalarTypeIdentification<signed char>();

template<>
int
GetVTKScalarTypeIdentification<unsigned char>();

//*******************************************************************
template< typename ElementType >
void
SetVTKImageDataScalarType( vtkImageData & dataset )
{
	dataset.SetScalarType( GetVTKScalarTypeIdentification< ElementType > );
}

/**
 * Fill instance of vtkImageData with copy of image passed as argument.
 * @param imageData Pointer to vtkImageData object. Must be valid!
 * @param image Instance of image, which should be converted to VTK representation.
 * @exception EImpossibleVTKConversion
 **/
template< typename ElementType >
void
FillVTKImageDataFromImageData( 
		vtkImageData					*imageData,
		const Images::ImageDataTemplate< ElementType >	&image 
		)
{
	size_t width, height, depth;
	vtkIdType IncX, IncY, IncZ;

	width	= image.GetDimensionInfo( 0 ).size;
	height	= image.GetDimensionInfo( 1 ).size;
	depth	= image.GetDimensionInfo( 2 ).size;	

	//imageData->SetSpacing(voxelsize.x, voxelsize.y, voxelsize.z);
	imageData->SetDimensions(width, height, depth);

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
	vtkImageData* imageData = vtkImageData::New();

	try {	
		FillVTKImageDataFromImageData( imageData, image );
	}
	catch(...) {
		//Free unused data and throw exception again.
		delete imageData;
		throw;
	}

	return imageData;
}


}/*namespace vtkIntegration*/
}/*namespace M4D*/

#endif /*_DATA_CONVERSION_H*/
