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
		//return AbstractImage::APtr();	
	}

	//We need something to work with, otherwise throw exception.
	if( dicomObjects->empty() ) {
		D_PRINT( "-----EMPTY DICOM OBJECTS SET -> THROWING EXCEPTION----" );
		throw EEmptyDicomObjSet();
		//return AbstractImage::APtr();	
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

	D_PRINT( "------ Image size		= " << imageSize );
	D_PRINT( "------ Element size	= " << elementSize );
	D_PRINT( "------ Slice stride	= " << sliceStride );
	D_PRINT( "------ Width			= " << width );
	D_PRINT( "------ Height			= " << height );
	D_PRINT( "------ Depth			= " << depth );

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

	/*D_PRINT( "---- Flushing DObjects to array" );

	//Copy each slice into image to its place.
	size_t i = 0;
	for( 
		Dicom::DcmProvider::DicomObjSet::iterator it = dicomObjects->begin();
		it != dicomObjects->end();
		++it, ++i
	   ) {
		   DL_PRINT( 8, "-------- DICOM object " << it->OrderInSet() << " is flushed.");
		//TODO check parameter type if it will stay unit16*.
		it->FlushIntoArray( (uint16*)dataArray + (sliceStride * i / *it->OrderInSet()* / ) );
	}*/
	


	D_PRINT( "-- Leaving CreateImageFromDICOM() - everything OK" );
	D_PRINT( LogDelimiter( '+' ) );
	//Finally build and return image object.
	return result;
}

void
ImageFactory::FlushDicomObjects(
		Dicom::DcmProvider::DicomObjSetPtr dicomObjects,
		uint8 			pixelSize, 
		size_t 			imageSize,
		size_t			stride,
		uint8			* dataArray
		)
{

}

}/*namespace Images*/
}/*namespace M4D*/
