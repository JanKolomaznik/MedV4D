#ifndef MESSAGE_HEADERS_H
#define MESSAGE_HEADERS_H

#include "netStreamImpl.h"

namespace M4D
{
namespace CellBE
{

#define BIG_ENDIAN 0
#define LITTLE_ENDIAN 1

static uint8
GetEndianess( void)
{
  uint16 tmp = 1; // for endian testing
  uint8 *ptr = (uint8 *)&tmp;
    
  if( ptr[0] == 1)
    return LITTLE_ENDIAN;
  else
    return BIG_ENDIAN;
}

///////////////////////////////////////////////////////////////////////////////
  
  /**
   *  This structures represents ID of a job. It consists of two main parts.
   *  The first 
   *  For details see ClinetJob::GenerateJobID method implementation.
   */
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

///////////////////////////////////////////////////////////////////////////////

struct PrimaryJobHeader
{
  PrimaryJobHeader() {}

  PrimaryJobHeader( const PrimaryJobHeader & b)
  {
    action = b.action;
    id = b.id;
    endian = b.endian;
    nexPartLength = b.nexPartLength;
  }

  uint8 action;

  JobID id;

  uint8 endian;

  uint16 nexPartLength;
  
  static void Serialize( PrimaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( PrimaryJobHeader) );

    s << h->action << h->id << GetEndianess() << h->nexPartLength;
  }

  static void Deserialize( PrimaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( PrimaryJobHeader) );

    s >> h->action >> h->id >> h->endian >> h->nexPartLength;
  }
};

///////////////////////////////////////////////////////////////////////////////

#include "dataPieceHeader.h"

static void
DataPieceHeaderSerialize( DataPieceHeader *h)
{
  NetStreamArrayBuf s( (uint8 *)h, sizeof( DataPieceHeader) );

  s << h->pieceSize;
}

static void
DataPieceHeaderDeserialize( DataPieceHeader *h)
{
  NetStreamArrayBuf s( (uint8 *)h, sizeof( DataPieceHeader) );

  s >> h->pieceSize;
}

///////////////////////////////////////////////////////////////////////////////

enum ResponseID
{
  RESPONSE_OK,
  RESPONSE_ERROR_IN_EXECUTION,
  RESPONSE_ERROR_IN_INPUT
};

struct ResponseHeader
{
  uint8 result;
  uint16 resultPropertiesLen;

  static void Serialize( ResponseHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( ResponseHeader) );

    s << h->result << h->resultPropertiesLen;
  }
  
  static void Deserialize( ResponseHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( ResponseHeader) );

    s >> h->result >> h->resultPropertiesLen;
  }
};

///////////////////////////////////////////////////////////////////////////////
  
}}
#endif

