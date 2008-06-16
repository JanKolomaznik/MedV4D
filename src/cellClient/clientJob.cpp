
#include "clientJob.h"

using namespace M4D::CellBE;
using namespace std;

uint32 ClientJob::lastID;

///////////////////////////////////////////////////////////////////////

ClientJob::ClientJob( bool isPersistent_) 
  : onComplete( NULL)
  , onError(NULL)
{
  jobID = ++lastID;
  m_isPersistent = isPersistent_;

  primHeader.id = jobID;
  primHeader.persistentJob = m_isPersistent;
  primHeader.action = (uint8) CREATE;
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::SerializeFiltersSetting( void)
{
  if( m_filters.empty() )
  {
    LOG( "Empty filter vector. Add some filter into job!");
    throw ExceptionBase( "Empty filter vector. Add some filter into job!");
  }

  for( FilterMap::iterator it=m_filters.begin(); it != m_filters.end(); it++)
  {
    it->second->Serialize( filterSettingsSerialized);
  }

  secHeader.filterSettStreamLen = (uint16) filterSettingsSerialized.size();
}

///////////////////////////////////////////////////////////////////////

FilterSetting *
ClientJob::GetFilter( std::string f)
{
  FilterMap::iterator tmp = m_filters.find( f);
  if( tmp == m_filters.end() )
    throw ExceptionBase("Not a filter in job");
  else
    return tmp->second;
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::AddFilter( std::string fID, FilterSetting *sett)
{
  m_filters.insert( FilterMap::value_type( fID, sett ) );
}

///////////////////////////////////////////////////////////////////////

void
ClientJob::Serialize( NetStream &s)
{
  primHeader.persistentJob = (uint8) m_isPersistent;
  primHeader.id = jobID;

  //primHeader.Serialize( s);
  //secHeader.Serialize( s);
}

///////////////////////////////////////////////////////////////////////

// TODO move to server part
void
ClientJob::DeSerialize( NetStream &s)
{
  //primHeader.DeSerialize( s);

  jobID = primHeader.id;
  m_isPersistent = primHeader.persistentJob == 1;
}

///////////////////////////////////////////////////////////////////////