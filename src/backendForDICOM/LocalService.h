#ifndef M4DDICOMLOCALSERVICE
#define M4DDICOMLOCALSERVICE

/**
 *  Implements searching and gettting functions to local FS dicom files
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

  struct Entry
  {
    std::string patID;
    std::string studyID;
    //std::string setID;

    bool operator< (const Entry &b) const
    {
      return (
        patID.compare( b.patID) < 0
        || studyID.compare( b.studyID) < 0
        );
    }
  };


  typedef std::set<std::string> SetIDsInStudy;
  typedef std::map<Entry, SetIDsInStudy> SetOfEntries;
  SetOfEntries m_setOfEntries;

  std::queue<boost::filesystem::path> m_mainQueue;

  std::string m_lastSearchDir;

	void Find( 
			DcmProvider::ResultSet &result,
      const std::string &path);

  void FindStudyInfo( 
    DcmProvider::StringVector &result,
      const std::string &patientID,
			const std::string &studyID);

  void GetImageSet(
      const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
      DcmProvider::DicomObjSet &result);

  void SolveDir( boost::filesystem::path & dirName,
    DcmProvider::ResultSet &result);

  void SolveFile( const std::string & fileName,
    const std::string & dirName,
    DcmProvider::ResultSet &result);

  void SolveDirGET( boost::filesystem::path & dirName,
    const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
    DcmProvider::DicomObjSet &result);

  void SolveFileGET( const std::string & fileName,
    const std::string &patientID,
		const std::string &studyID,
		const std::string &serieID,
    DcmProvider::DicomObjSet &result);
	
};

} // namespace
}

#endif

