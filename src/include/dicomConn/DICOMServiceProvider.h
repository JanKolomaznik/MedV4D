#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>
#include <map>
#include <string>
#include <boost/shared_ptr.hpp>


/**
 *  \file DICOMServiceProvider.h
 *  \brief Declares class for handling DICOM issues
 *  \author Vaclav Klecanda

 *  \mainpage
 *
 *  \section medv4d Project Medv4D
 *  Faculty of Mathematics and Physics\n
 *  Charles University, Czech Republic\n
 *
 *  \section Description
 *  This project ...
 *  
 *  \section Authors
 *  Jan Kolomaznik\n
 *  Attila Ulman\n
 *  Szabolcz Grof\n
 *  Vaclav Klecanda\n
 * 
 *
 * \addtogroup dicom DICOM management
 * @{
 *  \section DICOM
 *  DICOM is a standard describing creating, storing, manipulating and many more actions taken on medical data. For full description informations see (http://www.dclunie.com/dicom-status/status.html).
 *  That data are normally stored on DICOM server. There are many implementations of that server. Most common is PACS server. DICOM server implements searching, moving, storing functionalities. 
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

	/// This represents a DICOM file.
  /** DICOM file is structured file. Elements
   *  of that structure are pair of <tag id, tag value> ( for tags definitions see 
   *  DICOM doc ([ver]_05.pdf chapter 6.2) ). Each tag has defined also it value 
   *  representation (VR : see DICOM doc ([ver]_05.pdf chapter 6.2)) that is actualy 
   *  data type of an tag. Whole set of an pairs (data elements) is called dataSet. 
   *  In each data set is normally (but can be more) data element that stores actual data 
   *  of the file. That element's data has a special format and must be specialy encoded 
   *  (see DICOM doc ([ver]_05.pdf section 8). That special format is called DICOM stream.
   *  DicomObj can be retrieved from DICOM server,
   *  or loaded from local filesystem (disc, networkplace). Both ways through DcmProvider methods. When it is being 
   *  retrieved, Init function is called. This cause some basic information
   *  to be loaded from dataSet and OnLoaded callback is called.
   */
  class DicomObj
  {

  public:

    /// Typedef for callbacks for events
    typedef void (*ImageLoadedCallback)(void);

	  inline void SetLoadedCallback( ImageLoadedCallback c) 
		  { m_loadedCallBack = c; }

    /// Pointer to actual dataset container.
    /**   Void is used because of unimportance
     *  of files that use (and include) this file to know something about dataSet
     *  that is defined in DCMTK libraries that are quite bad organized and large.
     */
	  void *m_dataset;

    // support for loading & saving
    void *m_fileFormat;
    
	  //  Basic image information members.
    /// Retuerns size(in bytes) of one element
	  inline uint8 GetPixelSize( void) { return m_pixelSize; }
	  inline uint16 GetWidth( void) { return m_width; }
	  inline uint16 GetHeight( void) { return m_height; }
	  inline bool IsDataSigned( void) { return m_signed; }

	  void GetSliceThickness( float32 &f);
	  void GetPixelSpacing( float32 &horizSpacing, float32 &vertSpacing);
	  void GetSliceLocation( float32 &location);

    /// Converts from special dicom data stream
	  /** to steam of normal data types like uint16. See DICOM spec.
	   *	08-05pu.pdf (AnnexD).
	   *  \note Type T should corespond to size of pixel determined from GetPixelSize
	   *  method or NOTE: UNSPECIFIED behaviour will follow !!!!! .
     *  \param dest = destination buffer where to unpack the DICOM data stream.
	   */
	  template< typename T>
	  void FlushIntoArray( const T *dest);

	  void FlushIntoArrayNTID( void*dest, int elementTypeID );

	  /// Returns order number in set.
    /** according that images can be sorted
	   *  in order they were accuired by the mashine.
	   *  Currently SliceLocation tag is used.
	   */
	  inline int16 OrderInSet( void) const { return m_orderInSet; }

    /// For sorting issues.
    inline bool operator<( const DcmProvider::DicomObj &b) const
    {
      return OrderInSet() < b.OrderInSet();
    }

	  // load & save
	  void Load( const std::string &path);
	  void Save( const std::string &path);

	  /// Called when image arrive. 
    /** Inits basic info to member variables.
     *  These are returned via basic image information members.
	   */
	  void Init();

	  /// Gets values from data set container
    /**   (other that the basic) 
     */
	  void GetTagValue( uint16 group, uint16 tagNum, std::string &) ;
	  void GetTagValue( uint16 group, uint16 tagNum, int32 &);
	  void GetTagValue( uint16 group, uint16 tagNum, float32 &);
  	
	  ////////////////////////////////////////////////////////////

	  DicomObj();

  private:

	  enum Status {
		  Loaded,
		  Loading,
		  Failed,
	  };

	  // image info (basic set of information retrieved from data set.
	  // others are to be retrieved via GetTagValue method
	  uint16 m_width, m_height;
	  uint8 m_pixelSize;
	  bool m_signed;
	  int16 m_orderInSet;

	  Status m_status;

    // on loaded event callback
	  ImageLoadedCallback m_loadedCallBack;
  };

  // typedef for callbacks for events
  typedef void (*ImageLoadedCallback)(void);

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
	 *	@param modalities  - reference to vector of strings containing set 
	 *		of wanted modalities
	 *	@param dateFrom		 - ref. to string containing date in yyyyMMdd format
   *  $param dateTo     - the same for to
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
    ImageLoadedCallback on_loaded = NULL);

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

//#include "dicomConn/DICOMObject.h"

#endif

/**
 * @}
 */