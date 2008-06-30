#ifndef MESSAGE_HEADERS_H
#define MESSAGE_HEADERS_H

#include "netStream.h"

namespace M4D
{
namespace CellBE
{
  
#define IDLEN 12
struct JobID
{
  uint8 id[IDLEN];

  JobID() {}
  inline JobID(const JobID &b)
  {
    for( int i=0; i<IDLEN; i++)
      id[i] =  b.id[i];
  }

  bool operator < ( const JobID &b) const
  {
    for( int i=0; i<IDLEN; i++)
    {
      if( id[i] < b.id[i])
        return false;
    }
    return false;
  }

};

inline NetStream &operator<<( NetStream &s, const JobID &id)
{
  for( int i=0; i < IDLEN; i++)
    s << id.id[i];
  return s;
}

inline NetStream &operator>>( NetStream &s, JobID &id)
{
  for( int i=0; i < IDLEN; i++)
    s >> (uint8) id.id[i];
  return s;
}

inline std::ostream &operator<<( std::ostream &s, JobID &id)
{
  for( int i=0; i < IDLEN; i++)
    s << (char) id.id[i];
  return s;
}

///////////////////////////////////////////////////////////////////////

struct PrimaryJobHeader
{
  JobID id;
  
  uint8 action;

  uint8 dataSetType;

  uint8 endian;  
  
  static void Serialize( PrimaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( PrimaryJobHeader) );

    s << h->action << h->id << h->dataSetType;
#ifdef LITTLE_ENDIAN
    s << (uint8) 0;
#else
    s << (uint8) 1;
#endif
  }

  static void Deserialize( PrimaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( PrimaryJobHeader) );

    s >> h->action >> h->id >> h->dataSetType;
  }
};

///////////////////////////////////////////////////////////////////////

struct SecondaryJobHeader
{
  uint16 filterSettStreamLen;

  uint16 dataSetPropertiesLen;
  
  static void Serialize( SecondaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( SecondaryJobHeader) );

    s << h->filterSettStreamLen << h->dataSetPropertiesLen;
  }
  static void Deserialize( SecondaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( SecondaryJobHeader) );

    s >> h->filterSettStreamLen >> h->dataSetPropertiesLen;
  }  
};
  
}}
#endif