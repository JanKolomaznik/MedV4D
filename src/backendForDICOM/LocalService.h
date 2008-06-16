#ifndef M4DDICOMLOCALSERVICE
#define M4DDICOMLOCALSERVICE

/**
 *  Implements searching and gettting functions to local FS dicom files
 */
#include <string>
#include <set>

using namespace M4D::Dicom;

namespace M4D
{
namespace DicomInternal 
{

class LocalService
{
	friend class DcmProvider;

  struct Entry
  {
    std::string patID;
    std::string studyID;
    std::string setID;

    bool operator< (const Entry &b) const
    {
      return (
        patID.compare( b.patID) < 0
        || studyID.compare( b.studyID) < 0
        );
    }
  };

  typedef set<Entry> SetOfEntries;
  SetOfEntries m_setOfEntries;

  std::queue<boost::filesystem::path> m_mainQueue;

  std::string m_lastSearchDir;

	void Find( 
			DcmProvider::ResultSet &result,
      const std::string &path);

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
	
};

} // namespace
}

#endif

