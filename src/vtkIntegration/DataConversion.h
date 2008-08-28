#ifndef _DATA_CONVERSION_H
#define _DATA_CONVERSION_H

/// @defgroup vtkintegration VTK integration
/**
 *  @ingroup vtkintegration
 *  @file DataConversion.h
 *
 *  @addtogroup vtkintegration
 *  @{
 *  @section dataconv Data conversion
 *
 *  data has to be converted from our representation to VTK one
 */

#include "Imaging/Image.h"

#include "vtkImageData.h"
#include "ExceptionBase.h"

#include "Common.h"

#include <fstream>

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
FillVTKImageFromM4DImage( vtkImageData *vtkImage, const Imaging::AbstractImage &m4dImage );



template< typename ElementType >
int
GetVTKScalarTypeIdentification()
{
	return ConvertNumericTypeIDToVTKScalarType( GetNumericTypeID< ElementType >() );
	//throw EImpossibleVTKConversion();
}
/*
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
*/

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
		const Imaging::Image< ElementType, 3 >	&image 
		)
{
	/*size_t width	= image.GetDimensionInfo( 0 ).size;
	size_t height	= image.GetDimensionInfo( 1 ).size;
	size_t depth	= image.GetDimensionInfo( 2 ).size;*/	

	int	extent[6];
	for( size_t dim = 0; dim < 3; ++dim ) {
		const Imaging::DimensionExtents &dimExtents = image.GetDimensionExtents( dim );
		
		extent[2*dim]	= dimExtents.minimum;  
		extent[2*dim + 1] = dimExtents.maximum; 
		
		//_spacing[dim] = dimExtents.elementExtent;
	}

	vtkIdType IncX, IncY, IncZ;

	imageData->GetContinuousIncrements( extent, IncX, IncY, IncZ );
	ElementType *iPtr = (ElementType*)imageData->GetScalarPointer();

	//TODO delete
	//std::ofstream pomFile( "dump.txt" );
	//size_t i = 0;
	for(int idxZ = extent[4]; idxZ < extent[5]; ++idxZ)
	{
		for(int idxY = extent[2]; idxY < extent[3]; ++idxY)
		{
			for(int idxX = extent[0]; idxX < extent[1]; ++idxX)
			{
				//TODO delete
				//if( ++i < 2000 ) 
				//	pomFile << image.Get( idxX, idxY, idxZ ) << " ";
				
				*iPtr = image.GetElement( idxX, idxY, idxZ );
				++iPtr;
			}
			iPtr += IncY;
			
		}
		iPtr += IncZ;
	}
	//TODO delete
	//pomFile.close();
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
		const Imaging::Image< ElementType, 3 >& image )
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

/** @} */

#endif /*_DATA_CONVERSION_H*/
