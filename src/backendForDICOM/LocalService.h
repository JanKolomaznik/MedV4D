#ifndef M4DDICOMLOCALSERVICE
#define M4DDICOMLOCALSERVICE

/**
 *  Implements searching and gettting functions to local FS dicom files. It sequentialy loads data files in specified folder (and subfolders through queue), read ID info, based on that info and given filter inserts or not inserts (if matching found) record into result.
 *  Each search run writes the folder that is performed on, build structure of information that is used when aditional informations concerning data from the same run are required. 
 *  One run is quite expensive while loading each file is needed (there is no other way how to read required IDs). So it is sensitive how wide and deep is the subtree of the given folder.
 *  Maybe some timeouts will be required.
 *  All functions are private beacause are all called from friend class DcmProvider.
 */
#include <string>
#include <map>
#include <set>

using namespace M4D::Dicom;

namespace M4D
{
namespace DicomInternal 
{

class LocalService
{
	friend class M4D::Dicom::DcmProvider;

  /**
   *  Key entry of map build while search run. When additional information (serie info) is needed, just find in map is performed and found value is returned
   */
  struct Entry
  {
    std::string patID;
    std::string studyID;
    //std::string setID;

    inline bool operator< (const Entry &b) const
    {
      return (patID + studyID).compare( b.patID + b.studyID) < 0;
    }
  };

  void Reset(void);

  // informations structure containers definitions
  typedef std::set<DcmProvider::SerieInfo> SeriesInStudy;
  typedef std::map<Entry, SeriesInStudy> SetOfEntries;

  // actual structure instance
  SetOfEntries m_setOfEntries;

  // queue of remainig subfolders
  std::queue<boost::filesystem::path> m_mainQueue;

  // currently searched folder
  std::string m_lastSearchDir;

  Dicom::DcmProvider::ResultSet m_lastResultSet;

  // performs search run on given folder
	void Find( 
			DcmProvider::ResultSet &result,
      const std::string &path);

  // returns serie info based on build info structure
  void FindStudyInfo( 
    DcmProvider::SerieInfoVector &result,
      const std::string &patientID,
			const std::string &studyID);

  // performs search run and returns set of loaded data files (DicomObj)
  void GetImageSet(
      const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
      DcmProvider::DicomObjSet &result);

  // supporting functions to go on one folder or to solve single file
  void SolveDir( boost::filesystem::path & dirName,
    DcmProvider::ResultSet &result);
  // ...
  void SolveFile( const std::string & fileName,
    const std::string & dirName,
    DcmProvider::ResultSet &result);
  // ...
  void SolveDirGET( boost::filesystem::path & dirName,
    const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
    DcmProvider::DicomObjSet &result);
  // ...
  void SolveFileGET( const std::string & fileName,
    const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
    DcmProvider::DicomObjSet &result);
	
};

} // namespace
}

#endif

