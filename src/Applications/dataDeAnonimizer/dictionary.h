/**
 *  @ingroup datamodifier
 *  @file dictionary.h
 *  @brief Definition of Dictionary class
 *  @author Vaclav Klecanda
 */

#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <vector>
#include <set>
#include <map>
#include "entities.h"

/**
 *  @addtogroup datamodifier Data modifier( support utility)
 *  @{
 */

namespace M4D 
{
namespace DataModifier 
{

/// Dictionary of names loaded from cfg files.
/**
 *  Loads the cfg files. Contains methodes for generation
 *  of names (males & females) and born dates.
 */
class Dictionary
{
  // containers typedefs for already generated indexes
  typedef std::set<uint16> FornamesSet;
  typedef std::map<uint16, FornamesSet> SurnameContainer;

  SurnameContainer m_malesGenrated;
  SurnameContainer m_femalesGenrated;

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

  bool IsAlreadyGenerated( 
    uint16 foreNamePos, 
    uint16 sureNamePos, 
    SurnameContainer &cont);

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

/** @} */

#endif
