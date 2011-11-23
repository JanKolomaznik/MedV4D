/**
 *  @ingroup datamodifier
 *  @file dictionary.cpp
 *  @brief Implentation of Dictionary class
 *  @author Vaclav Klecanda
 */

#include <fstream>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "boost/date_time/gregorian/gregorian.hpp"


#include "MedV4D/Common.h"
#include "dictionary.h"

using namespace M4D::DataModifier;
using namespace boost::gregorian;
using namespace std;

/**
 *  @addtogroup datamodifier Data modifier( support utility)
 *  @{
 */
///////////////////////////////////////////////////////////////////////////////

#define FORENAMES_FILENAME "foreNames.cfg"
#define SURENAMES_FILENAME "sureNames.cfg"

Dictionary::Dictionary()
{
	// load dictionary from file
  {
    ifstream in( FORENAMES_FILENAME);
    if( ! in.good() )
    {
      LOG( FORENAMES_FILENAME << "Not found in cwd directory" );
      throw ExceptionBase( "cfg not found!");
    }
    LoadMaleFemaleFile( m_femaleForeNames, m_maleForeNames, in);
    in.close();

    ifstream inSur( SURENAMES_FILENAME);
    if( ! inSur.good() )
    {
      LOG( SURENAMES_FILENAME << "Not found in cwd directory" );
      throw ExceptionBase( "cfg not found!");
    }
    LoadMaleFemaleFile( m_femaleSureNames, m_maleSureNames, inSur);
    inSur.close();
  }

  srand( (unsigned int) time(0));
}

///////////////////////////////////////////////////////////////////////////////

void
Dictionary::LoadMaleFemaleFile( 
                        NameVector &females, 
                        NameVector &males, 
                        std::ifstream &file)
{
  string str;

  file >> str;
  if( str.compare( "[MALES]"))
  {
    LOG( "Missing [MALES] header!");
    throw ExceptionBase( "cfg malformed!");
  }
  file >> str;    // read next line
  // load males
  while( ! file.eof() && str.compare("[FEMALES]") )
  {
    males.push_back( str);
    file >> str;
  }
  if( ! file.eof() && str.compare( "[FEMALES]"))
  {
    LOG( "Missing [FEMALES] header!");
    throw ExceptionBase( "cfg malformed!");
  }
  file >> str;    // read next line
  // load females
  while( ! file.eof() && str.compare("[FEMALES]") )
  {
    females.push_back( str);
    file >> str;
  }
  females.push_back( str);
}

///////////////////////////////////////////////////////////////////////////////

void
Dictionary::FillPatientInfo( PatientInfo &info)
{
  int sex = (int) (rand() % 2);
  
  switch( sex)
  {
  case 0:
    GetMaleName( info.patName);
    info.patSex = "M";    
    break;

  case 1:
    GetFemaleName( info.patName);
    info.patSex = "F";
    break;
  }

  GetBornNumber( info.born);
}

///////////////////////////////////////////////////////////////////////////////
#define MAX_GENERATION_TRIES 10

void
Dictionary::GetFemaleName( string &name)
{
	uint16 foreNamePos, sureNamePos;  
  bool alreadyGenerated = true;
  uint16 tries = 0;

  do {
    foreNamePos = (uint16) (rand() % m_femaleForeNames.size());
    sureNamePos = (uint16) (rand() % m_femaleSureNames.size());
    tries ++;

    // look if this pair was already generated  
    alreadyGenerated = IsAlreadyGenerated( foreNamePos, 
      sureNamePos, m_femalesGenrated);

  } while( tries < MAX_GENERATION_TRIES && alreadyGenerated );

  if( tries == MAX_GENERATION_TRIES)
    LOG( "Max count of tries was reached. Means too little dictionary !!!!");

  name.append( m_femaleForeNames[foreNamePos]);
  name.append("_");
  name.append( m_femaleSureNames[sureNamePos]);
}

///////////////////////////////////////////////////////////////////////////////

void
Dictionary::GetMaleName( string &name)
{
	uint16 foreNamePos, sureNamePos;  
  bool alreadyGenerated = true;
  uint16 tries = 0;

  do {
    foreNamePos = (uint16) (rand() % m_maleForeNames.size());
    sureNamePos = (uint16) (rand() % m_maleSureNames.size());
    tries ++;

    // look if this pair was already generated  
    alreadyGenerated = IsAlreadyGenerated( foreNamePos, 
      sureNamePos, m_malesGenrated);

  } while( tries < MAX_GENERATION_TRIES && alreadyGenerated );

  if( tries == MAX_GENERATION_TRIES)
    LOG( "Max count of tries was reached. Means too little dictionary !!!!");

  name.append( m_maleForeNames[foreNamePos]);
  name.append("_");
  name.append( m_maleSureNames[sureNamePos]);
}

///////////////////////////////////////////////////////////////////////////////

void
Dictionary::GetBornNumber(std::string &bornNum)
{
  std::stringstream stream;
	  
  stream << (int) (1970 + ( rand() % (2008 - 1970)) );  // year
  stream << setfill('0') << setw(2) << (int) (1 + (rand() % 12));  // month
  stream << setw(2) << (int) (1 + (rand() % 28));  // day

  bornNum = stream.str();
}

///////////////////////////////////////////////////////////////////////////////

void
Dictionary::GetDateBetween( 
    const std::string &from, 
    const std::string &to,
    std::string &date)
{
  int d, m, y;

  LoadDate( from, d, m, y);
  boost::gregorian::date d1( y, greg_month( m), d);

  LoadDate( to, d, m, y);
  boost::gregorian::date d2( y, greg_month( m), d);

  boost::gregorian::days daysBetween = d2 - d1;
  int rndDate = rand() % daysBetween.get_rep().as_number();

  boost::gregorian::date_duration rndDur( rndDate);

  d1 += rndDur;

  date.clear();
  stringstream outStream(date);

  outStream << setfill('0') << setw(4) << d1.year() << 
    setw(2) << (int)d1.month() <<
    setw(2) << d1.day();

  date = outStream.str();
}

///////////////////////////////////////////////////////////////////////////////

void
Dictionary::LoadDate( const std::string &date, 
                     int &day, int &month, int &year)
{
  char buf[5];
  stringstream dateStr( date);
  stringstream tmpStr;

  dateStr.read( buf, 4);
  buf[4] = 0;
  tmpStr << buf;
  tmpStr >> year;
  tmpStr.clear();

  dateStr.read( buf, 2);
  buf[2] = 0;
  tmpStr << buf;
  tmpStr >> month;
  tmpStr.clear();

  dateStr.read( buf, 2);
  buf[2] = 0;
  tmpStr << buf;
  tmpStr >> day;
}

///////////////////////////////////////////////////////////////////////////////

bool
Dictionary::IsAlreadyGenerated( uint16 foreNamePos, uint16 sureNamePos,
                               SurnameContainer &cont)
{
  SurnameContainer::iterator it = cont.find( sureNamePos);
  if( it == cont.end() )
  {
    FornamesSet fornames;
    fornames.insert( foreNamePos);

    cont.insert( SurnameContainer::value_type(
      sureNamePos, fornames) );
    return false;
  }
  else
  {
    FornamesSet::iterator fit = it->second.find( foreNamePos);
    if( fit == it->second.end() )
    {
      it->second.insert( foreNamePos);
      return false;
    }
    else
      return true;
  }
}

///////////////////////////////////////////////////////////////////////////////

/** @} */