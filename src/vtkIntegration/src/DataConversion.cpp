/**
 *  @ingroup vtkintegration
 *  @file DataConversion.cpp
 *  @brief some brief
 */
#include "vtkIntegration/DataConversion.h"

namespace M4D
{
namespace vtkIntegration
{

int
ConvertNumericTypeIDToVTKScalarType( int NumericTypeID )
{
	switch( NumericTypeID ) {
	case NTID_VOID: 		return VTK_VOID; 
	case NTID_INT_8: 		return VTK_SIGNED_CHAR;
	case NTID_UINT_8:	 	return VTK_UNSIGNED_CHAR;
	case NTID_INT_16: 		return VTK_SHORT;
	case NTID_UINT_16:	 	return VTK_UNSIGNED_SHORT;
	case NTID_INT_32:		return VTK_INT;
	case NTID_UINT_32:	 	return VTK_UNSIGNED_INT;
	case NTID_INT_64: 		return VTK_LONG;
	case NTID_UINT_64:	 	return VTK_UNSIGNED_LONG;
	case NTID_FLOAT_32: 		return VTK_FLOAT;
	case NTID_FLOAT_64: 		return VTK_DOUBLE;
	default:
					throw EImpossibleVTKConversion();
	}

}

template< typename ElementType >
void
TryFillVTKImageFromM4DImage( vtkImageData *vtkImage, const Imaging::AbstractImage &m4dImage )
{
	const Imaging::Image<ElementType, 3>* castedImage = 
			dynamic_cast<const Imaging::Image<ElementType,3>*>( &m4dImage );

	if( castedImage == NULL ) {
		throw EImpossibleVTKConversion();
	}
	FillVTKImageDataFromImageData< ElementType >( vtkImage, *castedImage );
}

void
FillVTKImageFromM4DImage( vtkImageData *vtkImage, const Imaging::AbstractImage &m4dImage )
{
	DL_PRINT( 8, "FillVTKImageFromM4DImage(), element ID " << m4dImage.GetElementTypeID() );
	NUMERIC_TYPE_TEMPLATE_SWITCH_DEFAULT_MACRO(
		       	m4dImage.GetElementTypeID(), 
			throw EImpossibleVTKConversion(), 
			TryFillVTKImageFromM4DImage< TTYPE >( vtkImage, m4dImage )
			);
	/*switch ( m4dImage.GetElementTypeID() ) {
	case NTID_VOID:
		throw EImpossibleVTKConversion();
		break;
	case NTID_SIGNED_CHAR:
		TryFillVTKImageFromM4DImage< signed char >( vtkImage, m4dImage );
		break;
	case NTID_UNSIGNED_CHAR:
		TryFillVTKImageFromM4DImage< unsigned char >( vtkImage, m4dImage );
		break;
	case NTID_SHORT:
		TryFillVTKImageFromM4DImage< short >( vtkImage, m4dImage );
		break;
	case NTID_UNSIGNED_SHORT:
		TryFillVTKImageFromM4DImage< unsigned short >( vtkImage, m4dImage );
		break;
	case NTID_INT:
		TryFillVTKImageFromM4DImage< int >( vtkImage, m4dImage );
		break;
	case NTID_UNSIGNED_INT:
		TryFillVTKImageFromM4DImage< unsigned int >( vtkImage, m4dImage );
		break;
	case NTID_LONG:
		TryFillVTKImageFromM4DImage< long >( vtkImage, m4dImage );
		break;
	case NTID_UNSIGNED_LONG:
		TryFillVTKImageFromM4DImage< unsigned long >( vtkImage, m4dImage );
		break;
	case NTID_LONG_LONG:
		TryFillVTKImageFromM4DImage< long long >( vtkImage, m4dImage );
		break;
	case NTID_UNSIGNED_LONG_LONG:
		TryFillVTKImageFromM4DImage< unsigned long long >( vtkImage, m4dImage );
		break;
	case NTID_FLOAT:
		TryFillVTKImageFromM4DImage< float >( vtkImage, m4dImage );
		break;
	case NTID_DOUBLE:
		TryFillVTKImageFromM4DImage< double >( vtkImage, m4dImage );
		break;
	default:
		throw EImpossibleVTKConversion();
		break;
	}*/

	return;
}
/*
template<>
int
GetVTKScalarTypeIdentification<float>()
{
	return VTK_FLOAT;
}

template<>
int
GetVTKScalarTypeIdentification<double>()
{
	return VTK_DOUBLE;
}

template<>
int
GetVTKScalarTypeIdentification<int>()
{
	return VTK_INT;
}

template<>
int
GetVTKScalarTypeIdentification<unsigned int>()
{
	return VTK_UNSIGNED_INT;
}

template<>
int
GetVTKScalarTypeIdentification<long>()
{
	return VTK_LONG;
}

template<>
int
GetVTKScalarTypeIdentification<unsigned long>()
{
	return VTK_UNSIGNED_LONG;
}

template<>
int
GetVTKScalarTypeIdentification<short>()
{
	return VTK_SHORT;
}

template<>
int
GetVTKScalarTypeIdentification<unsigned short>()
{
	return VTK_UNSIGNED_SHORT;
}

template<>
int
GetVTKScalarTypeIdentification<signed char>()
{
	return VTK_SIGNED_CHAR;
}

template<>
int
GetVTKScalarTypeIdentification<unsigned char>()
{
	return VTK_UNSIGNED_CHAR;
}*/


}/*namespace vtkIntegration*/
}/*namespace M4D*/
