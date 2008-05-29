#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <vector>
#include "entities.h"

/**
 * Dictionary of names
 */
namespace M4D {
namespace DataModifier {

class Dictionary
{
  typedef std::vector<std::string> NameVector;

  NameVector m_femaleForeNames;
  NameVector m_maleForeNames;
  NameVector m_femaleSureNames;
  NameVector m_maleSureNames;
  
  // returns random name from distionary within param
  void GetMaleName( std::string &name);
  void GetFemaleName( std::string &name);
  void GetBornNumber( std::string &bornNum);

  void LoadMaleFemaleFile( 
    NameVector &females, NameVector &males, std::ifstream &file);

  void LoadDate( const std::string &date, int &day, int &month, int &year);

public:
  Dictionary();

  void FillPatientInfo( PatientInfo &info);

  void GetDateBetween( 
    const std::string &from, 
    const std::string &to,
    std::string &date);
};

}
}

#endif
