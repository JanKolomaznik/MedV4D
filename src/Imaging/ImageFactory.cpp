#include "ImageFactory.h"
#include "Debug.h"

namespace M4D
{
namespace Images
{
using namespace Dicom;


void
ImageFactory::PrepareElementArrayFromTypeID( 
		int 		typeId, 
		size_t 		imageSize, 
		uint8		*& dataArray 
		)
{
	/*switch ( pixelSize ) {
		case 1 :
			dataArray = (uint8 *) PrepareElementArray< uint8 >( imageSize );
			break;
		case 2 :
			dataArray = (uint8 *) PrepareElementArray< uint16 >( imageSize );
			break;
		case 4 :
			dataArray = (uint8 *) PrepareElementArray< uint32 >( imageSize );
			break;
		default :
			//Shouldn't reach this...
			D_PRINT( "Unhandled pixel size" );
			ASSERT( false );
	}*/

	//We will generate switch over common numerical types. For more see Common.h
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		typeId, dataArray = (uint8 *) PrepareElementArray< TTYPE >( imageSize ) );
}

AbstractImage*
ImageFactory::CreateImageFromDataAndTypeID(
		int 			typeId, 
		size_t 			imageSize, 
		uint8			* dataArray, 
		DimensionInfo		* info
		)
{
	AbstractImage*	image;
/*	switch ( pixelSize ) {
		case 1 :
			image = new ImageDataTemplate< uint8 >( dataArray, info, 3, imageSize );
			break;
		case 2 :
			image = new ImageDataTemplate< uint16 >( (uint16*)dataArray, info, 3, imageSize ); 
			break;
		case 4 :
			image = new ImageDataTemplate< uint32 >( (uint32*)dataArray, info, 3, imageSize );
			break;
		default :
			//Shouldn't reach this...
			D_PRINT( "Unhandled pixel size" );
			ASSERT( false );
	}*/

	//We will generate switch over common numerical types. For more see Common.h
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		typeId, image = new ImageDataTemplate< TTYPE >( (TTYPE*)dataArray, info, 3, imageSize ) );

	return image;
}


AbstractImage::APtr 
ImageFactory::CreateImageFromDICOM( DcmProvider::DicomObjSetPtr dicomObjects )
{
		D_PRINT( LogDelimiter( '*' ) );
		D_PRINT( "-- Entering CreateImageFromDICOM()" );
	
	//Do we have valid pointer to dicom object set??
	if( !dicomObjects  ) {
		D_PRINT( "-----WRONG DICOM OBJECTS SET POINTER -> THROWING EXCEPTION----" );
		throw EWrongPointer();	
	}

	//We need something to work with, otherwise throw exception.
	if( dicomObjects->empty() ) {
		D_PRINT( "-----EMPTY DICOM OBJECTS SET -> THROWING EXCEPTION----" );
		throw EEmptyDicomObjSet();	
	}

	D_PRINT( "---- DICOM OBJECT SET size = " << dicomObjects->size() );

	//TODO - check if all objects has same parameters.
	//TODO - consider getting parameters from first one.

	//Get parameters of final image.
	uint8 elementSize = (*dicomObjects)[0].GetPixelSize(); //in bytes
	bool sign = (*dicomObjects)[0].IsDataSigned(); 

	int elementTypeID = GetNTIDFromSizeAndSign( elementSize, sign );
	if( elementTypeID == NTID_VOID ) {
		throw EUnknowDataType( elementSize, sign );
	}

	size_t	width = (*dicomObjects)[0].GetWidth();
	size_t	height = (*dicomObjects)[0].GetHeight();
	size_t	depth = dicomObjects->size();

	size_t	sliceSize = width * height;	//Count of elements in one slice.
	size_t	imageSize = sliceSize * depth;	//Count of elements in whole image.

	uint8*	dataArray = NULL;
	
	//How many bytes is needed to skip between two slices.
	size_t sliceStride = elementSize * sliceSize;

		D_PRINT( "---- Preparing memory for data." );
	//Create array for image elements.
	PrepareElementArrayFromTypeID( 
			elementTypeID, 
			imageSize, 
			dataArray	/*output*/ 
		       	);

		D_PRINT( "------ Image size     = " << imageSize );
		D_PRINT( "------ Element size	= " << elementSize );
		D_PRINT( "------ Slice stride   = " << sliceStride );
		D_PRINT( "------ Width          = " << width );
		D_PRINT( "------ Height         = " << height );
		D_PRINT( "------ Depth          = " << depth );

	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1 );
	info[1].Set( height, width );
	info[2].Set( depth, width * height );

		D_PRINT( "---- Creating resulting image." );
	AbstractImage::APtr result( (AbstractImage*)
				CreateImageFromDataAndTypeID( elementTypeID, imageSize, dataArray, info ) 
				);
	

	//We now copy data from dicom objects to prepared array. 
	//TODO - will be asynchronous.
	FlushDicomObjects( dicomObjects, elementTypeID, imageSize, sliceStride, dataArray );

		D_PRINT( "-- Leaving CreateImageFromDICOM() - everything OK" );
		D_PRINT( LogDelimiter( '+' ) );
	//Finally return image object.
	return result;
}
template< typename ElementType >
void
FlushDicomObjectsHelper(
		Dicom::DcmProvider::DicomObjSetPtr	&dicomObjects,
		size_t 					imageSize,
		size_t					stride,
		uint8					* dataArray
		)
{
	//Copy each slice into image to its place.
	size_t i = 0;
	for( 
		Dicom::DcmProvider::DicomObjSet::iterator it = dicomObjects->begin();
		it != dicomObjects->end();
		++it, ++i
	   ) {
		   DL_PRINT( 8, "-------- DICOM object " << it->OrderInSet() << " is flushed.");
		//it->FlushIntoArray< ElementType >( (ElementType*)dataArray + (stride * i /*it->OrderInSet()*/ ) );
		it->FlushIntoArrayNTID( (ElementType*)dataArray + (stride * i), GetNumericTypeID<ElementType>() );
	}
}

void
ImageFactory::FlushDicomObjects(
		Dicom::DcmProvider::DicomObjSetPtr	&dicomObjects,
		int 					elementTypeID, 
		size_t 					imageSize,
		size_t					stride,
		uint8					* dataArray
		)
{
		D_PRINT( "---- Flushing DObjects to array" );
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		elementTypeID, 
		FlushDicomObjectsHelper< TTYPE >( 
				dicomObjects, imageSize, stride, dataArray )
	);
}

}/*namespace Images*/
}/*namespace M4D*/
