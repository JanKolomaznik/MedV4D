#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>
#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
#include "dicomConn/DcmObject.h"

/**
 *  @file DICOMServiceProvider.h
 *  @brief Declares class for handling DICOM issues
 *  @author Vaclav Klecanda

 *  @mainpage
 *
 *  @section medv4d Project Medv4D
 *  Faculty of Mathematics and Physics\n
 *  Charles University, Czech Republic\n
 *
 *  @section Description
 *  This project ...
 *  
 *  @section Authors
 *  - Jan Kolomaznik
 *  - Attila Ulman
 *  - Szabolcz Grof
 *  - Vaclav Klecanda
 * 
 *
 *  @addtogroup dicom DICOM management
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
private:
  // pointers to particular service instances. void pointers are used
  // to eliminate includes of lower level library headers. They are 
  // casted in place of usage to rigth type.
	void *m_findService;
	void *m_moveService;
  void *m_localService;

public:

	// TYPEDEFS ///////////////////////////////////////////////////////////

	/// Represents one row in table that shows found results.
	struct TableRow 
  {
    std::string patientID;
		std::string name;
    std::string birthDate;
    bool sex;
    //std::string accesion;
    std::string studyID;
    std::string date;	
    std::string time;			
    std::string modality;
    std::string description;    
    std::string referringMD;
    //std::string institution;
    //std::string location;
    //std::string server;
    //std::string availability;
    //std::string status;
    //std::string user;
	};

  /// Result set - Vector of table rows.
	typedef std::vector<TableRow> ResultSet;

  /// Contains all series' infos of one Study.
  struct SerieInfo
  {
    std::string id;
    std::string description;

    bool operator <( const SerieInfo &b) const
    {
      return (id + description).compare( b.id + b.description) < 0;
    }
  };

  /// Vector of SerieInfos
  typedef std::vector<SerieInfo> SerieInfoVector;

	// vector of M4DSetInfo
  typedef std::vector<std::string> StringVector;
	typedef std::map<std::string, StringVector> StudyInfo;

  /// Container for one serie of images
	typedef std::vector<DicomObj> DicomObjSet;
  /// shared pointer to DicomObjSet type
	typedef boost::shared_ptr< DicomObjSet > DicomObjSetPtr;

	// METHODs ////////////////////////////////////////////////////////////

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
	void Find( 
		DcmProvider::ResultSet &result,
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
  void LocalFind( 
		DcmProvider::ResultSet &result,
    const std::string &path);

	/**
   *  Find provides informations about patient and study. But there can be more series in one study. So this member returns ID of all series of given study (seriesInstanceUIDs). There is normally only one.
   */
	void FindStudyInfo(
		const std::string &patientID,
		const std::string &studyID,
		SerieInfoVector &info) ;

  /**
   *  The same as FindStudyInfo. Works with local filesystem.
   */
	void LocalFindStudyInfo(
		const std::string &patientID,
		const std::string &studyID,
		SerieInfoVector &info) ;

	/**
   *  The same as FindStudyInfo but gets even imageIDs. Rarely used.
   */
	void FindStudyAndImageInfo(
		const std::string &patientID,
		const std::string &studyID,
		StudyInfo &info) ;

	/**
   *  Finds all studies concerning given patient. Construct special query
   *  to DICOM server to retrieve all patient's studies.
   */
	void FindAllPatientStudies(  
		const std::string &patientID,
		ResultSet &result) ;

  /**
   *  Send C-MOVE request to DICOM server to retrieve specified image.
   *  Image has to be specified through all level of IDs (patient, study, set, image)
   */
	void GetImage(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		const std::string &imageID,
		DicomObj &object) ;

  /**
   *  Send C-MOVE request to retrieve all images in set.
   */
	void GetImageSet(
		const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		DicomObjSet &result,
    DicomObj::ImageLoadedCallback on_loaded = NULL);

  /**
   *  Retrieve images from local filesystem.
   */
  void LocalGetImageSet(
    const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
		DicomObjSet &result);

  // ctor, dtor
	DcmProvider();

  /// Parametrized ctor
  /** If blocking param is true (default ctor behaviour), all methodes
   *  using dicom server will be blocking. False means nonblocking.
   */
  DcmProvider( bool blocking);

	~DcmProvider();
};


}
}

#endif

/**
 * @}
 */