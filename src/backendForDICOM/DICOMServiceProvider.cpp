
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

#include "Common.h"

#include "dicomConn/DICOMServiceProvider.h"

#include "DicomAssoc.h"

#include "AbstractService.h"
#include "MoveService.h"
#include "FindService.h"
#include "LocalService.h"

using namespace M4D::DicomInternal;
using namespace M4D::Imaging;

namespace M4D
{
namespace Dicom 
{

///////////////////////////////////////////////////////////////////////

DcmProvider::DcmProvider( bool blocking)
  :m_findService(NULL)
  ,m_moveService(NULL)
  ,m_localService(NULL)
{
  m_findService = new FindService();
  ((FindService *)m_findService)->SetMode( blocking);

	m_moveService = new MoveService();
  ((MoveService *)m_moveService)->SetMode( blocking);

  m_localService = new LocalService();
}

///////////////////////////////////////////////////////////////////////

DcmProvider::DcmProvider()
  :m_findService(NULL)
  ,m_moveService(NULL)
  ,m_localService(NULL)
{
	m_findService = new FindService();
	m_moveService = new MoveService();
  m_localService = new LocalService();
}

///////////////////////////////////////////////////////////////////////

DcmProvider::~DcmProvider()
{
  if( m_findService != NULL)
	  delete ( (FindService *) m_findService);
  if( m_moveService != NULL)
	  delete ( (MoveService *) m_moveService);
  if( m_localService != NULL)
    delete ( (LocalService *) m_localService);
}

///////////////////////////////////////////////////////////////////////
// Theese function are only inline redirections to member functions
// of appropriate service objects

void
DcmProvider::Find( 
		DcmProvider::ResultSet &result,
    const std::string &patientForeName,
    const std::string &patientSureName,
    const std::string &patientID,
		const std::string &dateFrom,
		const std::string &dateTo,
    const std::string &referringMD,
    const std::string &description) 
{
	static_cast<FindService *>(m_findService)->FindForFilter( 
    result, patientForeName, patientSureName, patientID,
    dateFrom, dateTo, referringMD, description);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalFind( 
			DcmProvider::ResultSet &result,
      const std::string &path)
{
  static_cast<LocalService *>(m_localService)->Find( result, path);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalFindStudyInfo( 
      const std::string &patientID,
			const std::string &studyID,
      SerieInfoVector &info)
{
  static_cast<LocalService *>(m_localService)->FindStudyInfo( 
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
  static_cast<LocalService *>(m_localService)->GetImageSet( 
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
	static_cast<FindService *>(m_findService)->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyAndImageInfo(
		const std::string &patientID,
		const std::string &studyID,
		StudyInfo &info) 
{
	static_cast<FindService *>(m_findService)->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindAllPatientStudies(  
		const std::string &patientID,
		ResultSet &result) 
{
	static_cast<FindService *>(m_findService)->FindStudiesAboutPatient( 
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
	static_cast<MoveService *>(m_moveService)->MoveImage( 
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
	static_cast<MoveService *>(m_moveService)->MoveImageSet( 
		patientID, studyID, serieID, result, on_loaded);

  // sort the vector of images
  std::sort( result.begin(), result.end() );
}

///////////////////////////////////////////////////////////////////////

AbstractImage::Ptr 
DcmProvider::CreateImageFromDICOM( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
{
	//TODO exceptions
	AbstractImageData::APtr data = CreateImageDataFromDICOM( dicomObjects );

	AbstractImage *imagePtr = NULL;
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(
			data->GetElementTypeID(), imagePtr = new Image< TTYPE, 3 >( data ) );

	return AbstractImage::Ptr( imagePtr );
}

///////////////////////////////////////////////////////////////////////

AbstractImageData::APtr 
DcmProvider::CreateImageDataFromDICOM( 
		M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjects )
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
		AbstractImageData::APtr result( (AbstractImageData*)
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

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FlushDicomObjects(
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

///////////////////////////////////////////////////////////////////////

} // namespace
}
/** @} */

