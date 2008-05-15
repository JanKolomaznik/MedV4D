#include "ImageFactory.h"
#include "Debug.h"

namespace M4D
{
namespace Images
{
using namespace Dicom;


void
ImageFactory::PrepareElementArrayFromPixelSize( 
		DcmProvider::DicomObj::PixelSize 	pixelSize, 
		size_t 					imageSize, 
		uint8					*& dataArray, 
		unsigned short 				&  elementSize 
		)
{
	switch ( pixelSize ) {
		case DcmProvider::DicomObj::bit8 :
			dataArray = (uint8 *) PrepareElementArray< uint8 >( imageSize );
			elementSize = 1;
			break;
		case DcmProvider::DicomObj::bit16 :
			dataArray = (uint8 *) PrepareElementArray< uint16 >( imageSize );
			elementSize = 2;
			break;
		case DcmProvider::DicomObj::bit32 :
			dataArray = (uint8 *) PrepareElementArray< uint32 >( imageSize );
			elementSize = 4;
			break;
		default :
			//Shouldn't reach this...
			ASSERT( false );
	}
}

AbstractImage*
ImageFactory::CreateImageFromDataAndPixelType(
		DcmProvider::DicomObj::PixelSize 	pixelSize, 
		size_t 					imageSize, 
		uint8					* dataArray, 
		DimensionInfo				* info
		)
{
	AbstractImage*	image;
	switch ( pixelSize ) {
		case DcmProvider::DicomObj::bit8 :
			image = new ImageDataTemplate< uint8 >( dataArray, info, 3, imageSize );
			break;
		case DcmProvider::DicomObj::bit16 :
			image = new ImageDataTemplate< uint16 >( (uint16*)dataArray, info, 3, imageSize ); 
			break;
		case DcmProvider::DicomObj::bit32 :
			image = new ImageDataTemplate< uint32 >( (uint32*)dataArray, info, 3, imageSize );
			break;
		default :
			//Shouldn't reach this...
			ASSERT( false );
	}
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

	//Get parameters of final image.
	size_t	width = (*dicomObjects)[0].GetWidth();
	size_t	height = (*dicomObjects)[0].GetHeight();
	size_t	depth = dicomObjects->size();

	size_t	sliceSize = width * height;	//Count of elements in one slice.
	size_t	imageSize = sliceSize * depth;	//Count of elements in whole image.

	DcmProvider::DicomObj::PixelSize elementSizeType = (*dicomObjects)[0].GetPixelSize();
	unsigned short elementSize = 1; //in bytes, will be set to right value later.
	uint8*	dataArray = NULL;
	
	//How many bytes is needed to skip between two slices.
	size_t sliceStride = elementSize * sliceSize;

	D_PRINT( "---- Preparing memory for data." );
	

	//Create array for image elements.
	PrepareElementArrayFromPixelSize( 
			elementSizeType, 
			imageSize, 
			dataArray,	/*output*/ 
			elementSize	/*output*/
		       	);

	D_PRINT( "------ Image size		= " << imageSize );
	D_PRINT( "------ Element size	= " << elementSize );
	D_PRINT( "------ Slice stride	= " << sliceStride );
	D_PRINT( "------ Width			= " << width );
	D_PRINT( "------ Height			= " << height );
	D_PRINT( "------ Depth			= " << depth );

	

	D_PRINT( "---- Flushing DObjects to array" );

	//Copy each slice into image to its place.
	size_t i = 0;
	for( 
		Dicom::DcmProvider::DicomObjSet::iterator it = dicomObjects->begin();
		it != dicomObjects->end();
		++it, ++i
	   ) {
		   DL_PRINT( 8, "-------- DICOM object " << it->OrderInSet() << " is flushed.");
		//TODO check parameter type if it will stay unit16*.
		it->FlushIntoArray( (uint16*)dataArray + (sliceStride * i /*it->OrderInSet()*/ ) );
	}
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1 );
	info[1].Set( height, width );
	info[2].Set( depth, width * height );

	D_PRINT( "---- Creating resulting image." );

	AbstractImage::APtr result( (AbstractImage*)
				CreateImageFromDataAndPixelType( elementSizeType, imageSize, dataArray, info ) 
				);

	D_PRINT( "-- Leaving CreateImageFromDICOM() - everything OK" );
	D_PRINT( LogDelimiter( '+' ) );
	//Finally build and return image object.
	return result;
}

}/*namespace Images*/
}/*namespace M4D*/
