/**
 *  @file dictionary.h
 *  @brief Definition of PatientEntry class
 *  @author Vaclav Klecanda
 */

#ifndef ENTRY_CONTAINER_H
#define ENTRY_CONTAINER_H

#include <map>
#include "entities.h"
#include "dictionary.h"

/**
 *  @addtogroup datamodifier Data modifier( support utility)
 *  @{
 */

namespace M4D {
namespace DataModifier {

/// Patient entry - key element for patient map.
struct PatientEntry 
{
  std::string patID;
  std::string path;
  PatientEntry( const std::string &patID_, const std::string &path_)
    : patID( patID_), path( path_) {}

  bool operator ==(const PatientEntry &b) const
  {
    return ( ! patID.compare( b.patID) &&
      ! path.compare( b.path) );
  }

  bool operator <(const PatientEntry &b) const
  {
    int res = patID.compare( b.patID);
    if( res)
      return res > 0;
    else
    {
      res = path.compare( b.path);
      return res > 0;
    }
  }
};

///////////////////////////////////////////////////////////////////////

/// Class encapsulating containers
/**(info maps) containing information about
 *  found patients and studies.
 *  </p>
 */
class EntryContainer
{
  // Patient map.
  typedef std::map<PatientEntry, PatientInfo> PatientInfoMap;
  // Study map. Key - studyInstanceUID
  typedef std::map<std::string, StudyInfo> StudyInfoMap;

  PatientInfoMap m_patients;
  StudyInfoMap m_studies;

  Dictionary m_dict; // name dictionary

public:
  // opens file, find & and modify the information, save
  void SolveFile( const std::string &name, const std::string &path);

  // prints info map to specified stram
  void FlushMaps( std::ofstream &out);

  // time span for generation of study date
  std::string dateFrom, dateTo;

  // flag. Don't modify the files but inly read info and than flush
  bool infoOnly;

  EntryContainer() 
    : infoOnly( false) 
  {}
};

}
}
/** @} */

#endif
