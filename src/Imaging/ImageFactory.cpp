#include "ImageFactory.h"
#include "Debug.h"

namespace M4D
{
namespace Images
{
using namespace M4DDicom;


/**
 * @param pixelSize Enum, which sais how many bits is used per pixel.
 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
 * @param dataArray Output parameter - allocated array is returned in this.
 * @param elementSize Output parametr - sais how many bytes is used by one element.
 **/
static void
PrepareElementArrayFromPixelSize( 
		DcmProvider::DicomObj::PixelSize 	pixelSize, 
		size_t 					imageSize, 
		uint8					*& dataArray, 
		unsigned short 				&  elementSize 
		)
{
	switch ( pixelSize ) {
		case DcmProvider::DicomObj::bit8 :
			dataArray = static_cast< uint8* > PrepareElementArray< uint8 >( imageSize );
			elementSize = 1;
			break;
		case DcmProvider::DicomObj::bit16 :
			dataArray = static_cast< uint8* > PrepareElementArray< uint16 >( imageSize );
			elementSize = 2;
			break;
		case DcmProvider::DicomObj::bit32 :
			dataArray = static_cast< uint8* > PrepareElementArray< uint32 >( imageSize );
			elementSize = 4;
			break;
		default :
			//Shouldn't reach this...
			ASSERT( false );
	}
}

/**
 * @param pixelSize Enum, which sais how many bits is used per pixel.
 * @param imageSize How many elements of size 'pixelSize' will be stored in array.
 * @param dataArray Filled array of image elements.
 * @param info Filled dimension info array.
 **/
static AbstractImage*
CreateImageFromDataAndPixelType(
		DcmProvider::DicomObj::PixelSize 	pixelSize, 
		size_t 					imageSize, 
		uint8					* dataArray, 
		DimensionInfo				* info,
		)
{
	AbstractImage*	image;
	switch ( pixelSize ) {
		case DcmProvider::DicomObj::bit8 :
			image = new ImageDataTemplate< uint8 >( dataArray, info, 3, imageSize ) 
			break;
		case DcmProvider::DicomObj::bit16 :
			image = new ImageDataTemplate< uint16 >( dataArray, info, 3, imageSize ) 
			break;
		case DcmProvider::DicomObj::bit32 :
			image = new ImageDataTemplate< uint32 >( dataArray, info, 3, imageSize ) 
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
	//TODO better parameters checking.
	if( !dicomObjects || dicomObjects->size() == 0 ) {
		return AbstractImage::APtr( NULL );	
	}
	size_t	width = (*dicomObjects)[0].GetWidth();
	size_t	height = (*dicomObjects)[0].GetHeight();
	size_t	depth = dicomObjects->size();

	size_t	sliceSize = width * height;	//Count of elements in one slice.
	size_t	imageSize = sliceSize * depth;	//Count of elements in whole image.

	DcmProvider::DicomObj::PixelSize elementSizeType = (*dicomObjects)[0].GetPixelSize();
	unsigned short elementSize = 1; //in bytes
	uint8*	dataArray = NULL;

	//Create array for image elements.
	PrepareElementArrayFromPixelSize( 
			elementSizeType, 
			imageSize, 
			dataArray,	/*output*/ 
			elementSize	/*output*/
		       	);


	//How many bytes is needed to skip between two slices.
	size_t sliceStride = elementSize * sliceSize;	

	//Copy each slice into image to its place.
	for(
		M4DDicom::DcmProvider::DicomObjSet::iterator it = dicomObjects->begin();
		it != dicomObjects->end();
		++it
	   ) {
		it->FlushIntoArray( dataArray + (sliceStride * it->OrderInSet()) );
	}
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1 );
	info[1].Set( height, width );
	info[2].Set( depth, width * height );

	//Finally build and return image object.
	return AbstractImage::APtr( 
			CreateImageFromDataAndPixelType( elementSizeType, imageSize, dataArray, info ) 
			);
}

}/*namespace Images*/
}/*namespace M4D*/
