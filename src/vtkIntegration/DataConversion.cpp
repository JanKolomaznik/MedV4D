#include "DataConversion.h"

namespace M4D
{
namespace vtkIntegration
{

int
ConvertNumericTypeIDToVTKScalarType( int NumericTypeID )
{
	switch( NumericTypeID ) {
	case NTID_VOID: 		return VTK_VOID; 
	case NTID_SIGNED_CHAR: 		return VTK_SIGNED_CHAR;
	case NTID_UNSIGNED_CHAR: 	return VTK_UNSIGNED_CHAR;
	case NTID_SHORT: 		return VTK_SHORT;
	case NTID_UNSIGNED_SHORT: 	return VTK_UNSIGNED_SHORT;
	case NTID_INT: 			return VTK_INT;
	case NTID_UNSIGNED_INT: 	return VTK_UNSIGNED_INT;
	case NTID_LONG: 		return VTK_LONG;
	case NTID_UNSIGNED_LONG: 	return VTK_UNSIGNED_LONG;
	case NTID_LONG_LONG: 		return VTK_LONG_LONG;
	case NTID_UNSIGNED_LONG_LONG: 	return VTK_UNSIGNED_LONG_LONG;
	case NTID_FLOAT: 		return VTK_FLOAT;
	case NTID_DOUBLE: 		return VTK_DOUBLE;
	default:
					return VTK_VOID;
	}

}

}/*namespace vtkIntegration*/
}/*namespace M4D*/
