
/**
 *  @ingroup dicom
 *  @file DICOMServiceProvider.cpp
 *  @author Vaclav Klecanda
 */

#include <queue>

#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmnet/dimse.h>
#include <dcmtk/dcmnet/diutil.h>
#include <dcmtk/dcmdata/dcdeftag.h>

#include "boost/filesystem/path.hpp"

#include "common/Common.h"

#include "../DICOMServiceProvider.h"
#include "../DicomAssoc.h"
#include "../AbstractService.h"
#include "../MoveService.h"
#include "../FindService.h"
#include "../LocalService.h"

using namespace M4D::Dicom;
using namespace M4D::Imaging;

///////////////////////////////////////////////////////////////////////
FindService *g_findService;
LocalService g_localService;
MoveService *g_moveService;

bool DcmProvider::_useRemotePart;
///////////////////////////////////////////////////////////////////////

void
DcmProvider::Init(void)
{
	_useRemotePart = DicomAssociation::InitAddressContainer();
	if(_useRemotePart)
	{
		g_findService = new FindService();
		g_moveService = new MoveService();
	}
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::Shutdown(void)
{
	if(_useRemotePart)
	{
		delete g_findService;
		delete g_moveService;
	}
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::Find(
		ResultSet &result,
    const std::string &patientForeName,
    const std::string &patientSureName,
    const std::string &patientID,
		const std::string &dateFrom,
		const std::string &dateTo,
    const std::string &referringMD,
    const std::string &description)
{
	g_findService->FindForFilter(
    result, patientForeName, patientSureName, patientID,
    dateFrom, dateTo, referringMD, description);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalFind(
			ResultSet &result,
      const std::string &path)
{
	g_localService.Find( result, path);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalFindStudyInfo(
      const std::string &patientID,
			const std::string &studyID,
      SerieInfoVector &info)
{
	g_localService.FindStudyInfo(
    info, patientID, studyID);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalGetImageSet(
      const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
			DicomObjSet &result)
{
	g_localService.GetImageSet(
    patientID, studyID, serieID, result);

  // sort the vector of images
  std::sort( result.begin(), result.end() );
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyInfo(
    const std::string &patientID,
		const std::string &studyID,
		SerieInfoVector &info)
{
	g_findService->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyAndImageInfo(
		const std::string &patientID,
		const std::string &studyID,
		StudyInfo &info)
{
	g_findService->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindAllPatientStudies(
		const std::string &patientID,
		ResultSet &result)
{
	g_findService->FindStudiesAboutPatient(
		patientID, result);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::GetImage(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		const std::string &imageID,
		DicomObj &object)
{
	g_moveService->MoveImage(
		patientID, studyID, serieID, imageID, object);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::GetImageSet(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		DicomObjSet &result,
    DicomObj::ImageLoadedCallback on_loaded)
{
	g_moveService->MoveImageSet(
		patientID, studyID, serieID, result, on_loaded);

  // sort the vector of images
  std::sort( result.begin(), result.end() );
}

///////////////////////////////////////////////////////////////////////

AImage::Ptr
DcmProvider::CreateImageFromDICOM( DicomObjSetPtr dicomObjects )
{
	//TODO exceptions
	AImageData::APtr data = CreateImageDataFromDICOM( dicomObjects );

	AImage *imagePtr = NULL;
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
			data->GetElementTypeID(), imagePtr = new Image< TTYPE, 3 >( data ) );

	return AImage::Ptr( imagePtr );
}

///////////////////////////////////////////////////////////////////////
struct DicomObjectComparatorPosition {
  bool operator() (const DicomObj &a, const DicomObj &b) 
  { 
	  float32 /*x1, y1,*/ z1;
	  float32 /*x2, y2,*/ z2;
	  //a.GetImagePosition( x1, y1, z1 );
	  //b.GetImagePosition( x2, y2, z2 );
	  //return a.OrderInSet() < b.OrderInSet();
	  a.GetSliceLocation( z1 );
	  b.GetSliceLocation( z2 );
	  return z1 < z2;
  }
};



AImageData::APtr
DcmProvider::CreateImageDataFromDICOM(
		DicomObjSetPtr dicomObjects )
{
		D_PRINT( LogDelimiter( '*' ) );
		D_PRINT( "-- Entering CreateImageFromDICOM()" );

	//Do we have valid pointer to dicom object set??
	if( !dicomObjects  ) {
		D_PRINT( "-----WRONG DICOM OBJECTS SET POINTER -> THROWING EXCEPTION----" );
		throw ImageFactory::EWrongPointer();
	}

	//We need something to work with, otherwise throw exception.
	if( dicomObjects->empty() ) {
		D_PRINT( "-----EMPTY DICOM OBJECTS SET -> THROWING EXCEPTION----" );
		throw ImageFactory::EEmptyDicomObjSet();
	}

	// first we have sort the images into right order !
	std::sort(dicomObjects->begin(), dicomObjects->end(), DicomObjectComparatorPosition() );

		D_PRINT( "---- DICOM OBJECT SET size = " << dicomObjects->size() );


	//TODO - check if all objects has same parameters.
	//TODO - consider getting parameters from first one.

	uint16 elementSize = (*dicomObjects)[0].GetPixelSize(); //in bytes
	bool sign = (*dicomObjects)[0].IsDataSigned();

	int elementTypeID = GetNTIDFromSizeAndSign( elementSize, sign );
	if( elementTypeID == NTID_VOID ) {
		throw ImageFactory::EUnknowDataType( elementSize, sign );
	}
	//Input tests finished. ------------------------------------------------------

	try
	{
		//Get parameters of final image.
		uint32	width	= (*dicomObjects)[0].GetWidth();
		uint32	height	= (*dicomObjects)[0].GetHeight();
		uint32	depth	= (uint32)dicomObjects->size();
		//Get extents of voxel.
		float32 voxelWidth = 1.0;
		float32 voxelHeight = 1.0;
		float32 voxelDepth = 1.0;
		(*dicomObjects)[0].GetPixelSpacing( voxelWidth, voxelHeight );
		//(*dicomObjects)[0].GetSliceThickness( voxelDepth );
		float32 tmp1 = 0.0;
		float32 tmp2 = 1.0;
		(*dicomObjects)[0].GetSliceLocation( tmp1 );
		(*dicomObjects)[1].GetSliceLocation( tmp2 );
		voxelDepth = Abs( tmp1 - tmp2 );

		if(voxelWidth <= 0.0f) voxelWidth = 1.0f;
		if(voxelHeight <= 0.0f) voxelHeight = 1.0f;
		if(voxelDepth <= 0.0f) voxelDepth = 1.0f;

		uint32	sliceSize = width * height;	//Count of elements in one slice.
		uint32	imageSize = sliceSize * depth;	//Count of elements in whole image.

		uint8*	dataArray = NULL;

		//How many bytes is needed to skip between two slices.
		uint32 sliceStride = elementSize * sliceSize;

			D_PRINT( "---- Preparing memory for data." );
		//Create array for image elements.
		ImageFactory::PrepareElementArrayFromTypeID(
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
		AImageData::APtr result( (AImageData*)
				ImageFactory::CreateImageFromDataAndTypeID( elementTypeID, imageSize, dataArray, info )
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

///////////////////////////////////////////////////////////////////////

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
		DicomObjSetPtr	&dicomObjects,
		uint32 					imageSize,
		uint32					stride,
		uint8					* dataArray
		)
{
	//Copy each slice into image to its place.
	uint32 i = 0;
	for(
		DicomObjSet::iterator it = dicomObjects->begin();
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

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FlushDicomObjects(
		DicomObjSetPtr	&dicomObjects,
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

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LoadSerieThatFileBelongsTo(const std::string &fileName,
		const std::string &folder, DicomObjSet &result)
{
	DicomObj o;
	o.Load(fileName);

	std::string seriesUID;
	o.GetTagValue(0x0020, 0x000e, seriesUID);
	std::string studyUID;
	o.GetTagValue(0x0020, 0x000d, studyUID);
	std::string patientID;
	o.GetTagValue(0x0010, 0x0020, patientID);

	g_localService.GetSeriesFromFolder(
			folder, patientID, studyUID, seriesUID, result);
}

///////////////////////////////////////////////////////////////////////

/** @} */

