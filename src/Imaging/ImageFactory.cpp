#include "ImageFactory.h"
#include "Debug.h"

namespace M4D
{
namespace Images
{
using namespace M4DDicom;


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
	//TODO better parameters checking.
	if( !dicomObjects || dicomObjects->size() == 0 ) {
		return AbstractImage::APtr();	
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
		//TODO check parameter type if it will stay unit16*.
		it->FlushIntoArray( (uint16*)dataArray + (sliceStride * it->OrderInSet()) );
	}
	
	//Preparing informations about dimensionality.
	DimensionInfo *info = new DimensionInfo[ 3 ];
	info[0].Set( width, 1 );
	info[1].Set( height, width );
	info[2].Set( depth, width * height );

	//Finally build and return image object.
	return AbstractImage::APtr( (AbstractImage*)
			CreateImageFromDataAndPixelType( elementSizeType, imageSize, dataArray, info ) 
			);
}

}/*namespace Images*/
}/*namespace M4D*/
