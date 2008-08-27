#include "Imaging/ImageFactory.h"
#include "Debug.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging
{
using namespace Dicom;


void
ImageFactory::PrepareElementArrayFromTypeID( 
		int 		typeId, 
		uint32 		imageSize, 
		uint8		*& dataArray 
		)
{
	//We will generate switch over common numerical types. For more see Common.h
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		typeId, dataArray = (uint8 *) PrepareElementArray< TTYPE >( imageSize ) );
}

AbstractImageData*
ImageFactory::CreateImageFromDataAndTypeID(
		int 			typeId, 
		uint32 			imageSize, 
		uint8			* dataArray, 
		DimensionInfo		* info
		)
{
	AbstractImageData*	image;

	//We will generate switch over common numerical types. For more see Common.h
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		typeId, image = new ImageDataTemplate< TTYPE >( (TTYPE*)dataArray, info, 3, imageSize ) );

	return image;
}

AbstractImage::AImagePtr 
ImageFactory::CreateImageFromDICOM( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
{
	//TODO exceptions
	AbstractImageData::APtr data = ImageFactory::CreateImageDataFromDICOM( dicomObjects );

	AbstractImage *imagePtr = NULL;
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
			data->GetElementTypeID(), imagePtr = new Image< TTYPE, 3 >( data ) );

	return AbstractImage::AImagePtr( imagePtr );
}

AbstractImageData::APtr 
ImageFactory::CreateImageDataFromDICOM( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
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

	uint16 elementSize = (*dicomObjects)[0].GetPixelSize(); //in bytes
	bool sign = (*dicomObjects)[0].IsDataSigned(); 

	int elementTypeID = GetNTIDFromSizeAndSign( elementSize, sign );
	if( elementTypeID == NTID_VOID ) {
		throw EUnknowDataType( elementSize, sign );
	}
	//Input tests finished. ------------------------------------------------------
	
	try 
	{
		//Get parameters of final image.
		uint32	width	= (*dicomObjects)[0].GetWidth();
		uint32	height	= (*dicomObjects)[0].GetHeight();
		uint32	depth	= dicomObjects->size();
		//Get extents of voxel.
		float32 voxelWidth = 1.0;
		float32 voxelHeight = 1.0;
		float32 voxelDepth = 1.0;
		(*dicomObjects)[0].GetPixelSpacing( voxelWidth, voxelHeight );
		(*dicomObjects)[0].GetSliceThickness( voxelDepth );

		uint32	sliceSize = width * height;	//Count of elements in one slice.
		uint32	imageSize = sliceSize * depth;	//Count of elements in whole image.

		uint8*	dataArray = NULL;
		
		//How many bytes is needed to skip between two slices.
		uint32 sliceStride = elementSize * sliceSize;

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
			D_PRINT( "-------- Voxel width  = " << voxelWidth );
			D_PRINT( "-------- Voxel height = " << voxelHeight );
			D_PRINT( "-------- Voxel depth  = " << voxelDepth );

		//Preparing informations about dimensionality.
		DimensionInfo *info = new DimensionInfo[ 3 ];
		info[0].Set( width, 1, voxelWidth);
		info[1].Set( height, width, voxelHeight );
		info[2].Set( depth, width * height, voxelDepth );

			D_PRINT( "---- Creating resulting image." );
		AbstractImageData::APtr result( (AbstractImageData*)
					CreateImageFromDataAndTypeID( elementTypeID, imageSize, dataArray, info ) 
					);
		

		//We now copy data from dicom objects to prepared array. 
		//TODO - will be asynchronous.
	 		D_PRINT( "---- Array start = " << (unsigned int*)dataArray << " array end = " << (unsigned int*)(dataArray + imageSize*elementSize) );
		FlushDicomObjects( dicomObjects, elementTypeID, imageSize, sliceStride, dataArray );

			D_PRINT( "-- Leaving CreateImageFromDICOM() - everything OK" );
			D_PRINT( LogDelimiter( '+' ) );
		//Finally return image object.
		return result;
	}
	catch ( ... ) {
		LOG( "Exception in CreateImageDataFromDICOM()" );
		throw;
	}
}

//TODO - improve this function
/** 
 * @param dicomObjects Set of DICOM objects which are flushed into the array.
 * @param imageSize How many elements of size 'pixelSize' can be stored in array.
 * @param stride Number of BYTES!!! used per one object flush (size of one layer in bytes).
 * @param dataArray Array to be filled from dicom objects. Must be allocated!!!
 * @exception EWrongArrayForFlush Thrown when NULL array passed, or imageSize is less than
 * space needed for flushing all dicom objects.
 **/
template< typename ElementType >
void
FlushDicomObjectsHelper(
		M4D::Dicom::DcmProvider::DicomObjSetPtr	&dicomObjects,
		uint32 					imageSize,
		uint32					stride,
		uint8					* dataArray
		)
{
	//Copy each slice into image to its place.
	uint32 i = 0;
	for( 
		Dicom::DcmProvider::DicomObjSet::iterator it = dicomObjects->begin();
		it != dicomObjects->end();
		++it, ++i
	   ) {
		uint8 *arrayPosition = dataArray + (stride * i);
		//uint8 *arrayPosition = dataArray + (stride * (it->OrderInSet()-1));
		//it->FlushIntoArray< ElementType >( (ElementType*)dataArray + (stride * i /*it->OrderInSet()*/ ) );

		//if( it->OrderInSet() <= 0 || it->OrderInSet() > dicomObjects->size() ) {
		//	throw ImageFactory::EWrongDICOMObjIndex();
		//}

		   DL_PRINT( 8, "-------- DICOM object " << it->OrderInSet() << " flushing to : " << (unsigned int*)arrayPosition );

		it->FlushIntoArrayNTID( (void*)arrayPosition, GetNumericTypeID<ElementType>() );
	}
}

void
ImageFactory::FlushDicomObjects(
		M4D::Dicom::DcmProvider::DicomObjSetPtr	&dicomObjects,
		int 					elementTypeID, 
		uint32 					imageSize,
		uint32					stride,
		uint8					* dataArray
		)
{
		D_PRINT( "---- Flushing DObjects to array" );
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		elementTypeID, 
		FlushDicomObjectsHelper< TTYPE >( dicomObjects, imageSize, stride, dataArray )
	);
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */