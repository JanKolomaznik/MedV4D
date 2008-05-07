#ifndef _DATA_CONVERSION_H
#define _DATA_CONVERSION_H

#include "ImageDataTemplate.h"

#include "vtkImageData.h"
#include "ExceptionBase.h"

#include "Common.h"

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

void
FillVTKImageFromM4DImage( vtkImageData *vtkImage, Images::AbstractImage::APtr m4dImage );



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
	size_t width	= image.GetDimensionInfo( 0 ).size;
	size_t height	= image.GetDimensionInfo( 1 ).size;
	size_t depth	= image.GetDimensionInfo( 2 ).size;	

	vtkIdType IncX, IncY, IncZ;
	
	int	extent[6];
	extent[0] = 0; extent[1] = width - 1;
	extent[2] = 0; extent[3] = height - 1;
	extent[4] = 0; extent[5] = depth - 1;

	imageData->GetContinuousIncrements( extent, IncX, IncY, IncZ );
	ElementType *iPtr = (ElementType*)imageData->GetScalarPointer();

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
		SetVTKImageDataScalarType< ElementType >( *imageData );

		//TODO - prepare DataSet

		FillVTKImageDataFromImageData( imageData, image );
	}
	catch(...) {
		//Unallocate unused data and throw exception again.
		delete imageData;
		throw;
	}

	return imageData;
}


}/*namespace vtkIntegration*/
}/*namespace M4D*/

#endif /*_DATA_CONVERSION_H*/
