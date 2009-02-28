#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

/**
 *  @ingroup dicom
 *  @file DICOMServiceProvider.h
 *  @brief Declares class for handling DICOM issues
 *  @author Vaclav Klecanda
 */

#include <vector>
#include <map>
#include <string>
#include "Common.h"
#include "structures.h"
#include "Imaging/ImageFactory.h"
#include "DcmObject.h"



/**
 *  @mainpage
 *
 *  @section medv4d Project Medv4D
 *  Faculty of Mathematics and Physics\n
 *  Charles University, Czech Republic\n
 *  www.mff.cuni.cz
 *
 *  @section desc Description
 *  Goal of this project is to create framework for easy development of
 *	user-friendly medical aplications. Suppose processing of 2D, 3D even 4D
 *	(for ex. time series), with input format DICOM storing on DICOM data server,
 *	parallel implementation of difficult algorithms and platform independence.
 *	It has to eliminate developing phases like DICOM files handling, GUI parts
 *	programing. User (programmer) can focus to write only a filter that represents
 *	the algorithm that he is interersted in.
 *  
 *  @section authors Authors
 *  - Jan Kolomaznik
 *  - Attila Ulman
 *  - Szabolcz Grof
 *  - Vaclav Klecanda
 * 
 *
 *  @defgroup dicom DICOM management
 *  @{
 *  @section DICOM
 *  DICOM is a standard describing creating, storing, manipulating and many
 *  more actions taken on medical data. For full description informations see
 *  (http://www.dclunie.com/dicom-status/status.html).
 *  That data are normally stored on DICOM server. There are many implementations
 *  of that server. Most common is PACS server. DICOM server implements searching,
 *  moving, storing functionalities.
 *  
 */

namespace M4D
{
namespace Dicom 
{

/// This class is gate to DICOM world.
/**
 *  This class provides following DICOM operations to upper layers: C-FIND,
 *  C-MOVE, C-STORE through function members that reflect most commonly 
 *  taken tasks.
 */
class DcmProvider
{
public:
	// METHODs ////////////////////////////////////////////////////////////
	
	/**
	 * Creates image from given dicomObject set.
	 * @param dicomObjects Given set of dicom objects.
	 * @return Smart pointer to created image.
	 * @exception ImageFactory::EWrongPointer Thrown when passed pointer isn't valid.
	 * @exception ImageFactory::EEmptyDicomObjSet Thrown when empty set passed.
	 * @exception ImageFactory::EUnknowDataType Thrown when type for element with 
	 * parameters from dicomObject doesn't exist.
	 **/
	static M4D::Imaging::AbstractImage::Ptr 
	CreateImageFromDICOM( DicomObjSetPtr dicomObjects );

	/**
	 * Creates image from given dicomObject set.
	 * @param dicomObjects Given set of dicom objects.
	 * @return Smart pointer to created image.
	 * @exception ImageFactory::EWrongPointer Thrown when passed pointer isn't valid.
	 * @exception ImageFactory::EEmptyDicomObjSet Thrown when empty set passed.
	 * @exception ImageFactory::EUnknowDataType Thrown when type for element with 
	 * parameters from dicomObject doesn't exist.
	 **/
	static M4D::Imaging::AbstractImageData::APtr 
	CreateImageDataFromDICOM( DicomObjSetPtr dicomObjects );
	
	//TODO - make this function asynchronous. Add locking of array in image.
	/**
	 * @param dicomObjects Set of dicom objects, which will be flushed into array.
	 * @param elementTypeID Type of stored elements.
	 * @param imageSize How many elements of size 'pixelSize' can be stored in array.
 	 * @param stride Number of BYTES!!! used per one object flush (size of one layer in bytes).
	 * @param dataArray Array to be filled from dicom objects. Must be allocated!!!
	 * @exception EWrongArrayForFlush Thrown when NULL array passed, or imageSize is less than
	 * space needed for flushing all dicom objects.
	 **/
	static void
	FlushDicomObjects(
		DicomObjSetPtr	&dicomObjects,
		int		 			elementTypeID, 
		uint32 					imageSize,
		uint32					stride,
		uint8					* dataArray
		);

	/// Send C-FIND request to DICOM server.
  /** Based on given filter params:
	 *	@param result - reference to result set object
	 *	@param patientForeName - reference to string containing fore name
   *  @param patientSureName - same with sure name
	 *	@param patientID   -   patient ID search mask
	 *	@param dateFrom		 - ref. to string containing date in yyyyMMdd format
   *  @param dateTo     - the same for to
   *	@param referringMD  - referring medician name
   *	@param description  - item decsription
	 */
	static void Find( 
		ResultSet &result,
    const std::string &patientForeName,
    const std::string &patientSureName,
    const std::string &patientID,
		const std::string &dateFrom,
		const std::string &dateTo,
    const std::string &referringMD,
    const std::string &description) ;

  /**
   *  Search given path recursively.
   *  @param result - result set containing results
   *  @param path   - given path
   */
	static void LocalFind( 
		ResultSet &result,
    const std::string &path);

	/**
   *  Find provides informations about patient and study. But there can be more series in one study. So this member returns ID of all series of given study (seriesInstanceUIDs). There is normally only one.
   */
	static void FindStudyInfo(
		const std::string &patientID,
		const std::string &studyID,
		SerieInfoVector &info) ;

  /**
   *  The same as FindStudyInfo. Works with local filesystem.
   */
	static void LocalFindStudyInfo(
		const std::string &patientID,
		const std::string &studyID,
		SerieInfoVector &info) ;

	/**
   *  The same as FindStudyInfo but gets even imageIDs. Rarely used.
   */
	static void FindStudyAndImageInfo(
		const std::string &patientID,
		const std::string &studyID,
		StudyInfo &info) ;

	/**
   *  Finds all studies concerning given patient. Construct special query
   *  to DICOM server to retrieve all patient's studies.
   */
	static void FindAllPatientStudies(  
		const std::string &patientID,
		ResultSet &result) ;

  /**
   *  Send C-MOVE request to DICOM server to retrieve specified image.
   *  Image has to be specified through all level of IDs (patient, study, set, image)
   */
	static void GetImage(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		const std::string &imageID,
		DicomObj &object) ;

  /**
   *  Send C-MOVE request to retrieve all images in set.
   */
	static void GetImageSet(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		DicomObjSet &result,
    DicomObj::ImageLoadedCallback on_loaded = NULL);

  /**
   *  Retrieve images from local filesystem.
   */
	static void LocalGetImageSet(
    const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		DicomObjSet &result);
	
	static DicomObjSet
	LoadSerieThatFileBelongsTo(const std::string &fileName);
};


}
}

#endif

/** @} */

