#ifndef ENTRY_CONTAINER_H
#define ENTRY_CONTAINER_H

#include <map>
#include "entities.h"
#include "dictionary.h"

/**
 * Dictionary of names
 */
namespace M4D {
namespace DataModifier {

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

class EntryContainer
{
  typedef std::map<PatientEntry, PatientInfo> PatientInfoMap;
  typedef std::map<std::string, StudyInfo> StudyInfoMap;

  PatientInfoMap m_patients;
  StudyInfoMap m_studies;

  Dictionary m_dict;

public:
  // opens file, find accod
  void SolveFile( const std::string &name, const std::string &path);

  void FlushMaps( std::ofstream &out);

  std::string dateFrom, dateTo;
  bool infoOnly;  // dont modify the files but inly read info and than flush

  EntryContainer() 
    : infoOnly( false) 
  {}
};

}
}

#endif
