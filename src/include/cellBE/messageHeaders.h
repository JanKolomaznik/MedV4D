/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file messageHeaders.h 
 * @{ 
 **/

#ifndef MESSAGE_HEADERS_H
#define MESSAGE_HEADERS_H

#include "cellBE/netStreamImpl.h"

namespace M4D
{
namespace CellBE
{

enum Endianness {
	End_BIG_ENDIAN = 0,
	End_LITTLE_ENDIAN = 1
};

static uint8
GetEndianess( void)
{
  uint16 tmp = 1; // for endian testing
  uint8 *ptr = (uint8 *)&tmp;
    
  if( ptr[0] == 1)
    return End_LITTLE_ENDIAN;
  else
    return End_BIG_ENDIAN;
}

///////////////////////////////////////////////////////////////////////////////
  
  /**
   *  This structures represents ID of a job. It consists of two main parts.
   *  The first 
   *  For details see ClinetJob::GenerateJobID method implementation.
   */
static const int IDLEN = 12;
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
    uint64 mine = 0;
    uint64 theirs = 0;
    for( int i=0; i<IDLEN; i++)
    {
      mine += id[i];
      theirs += b.id[i];
    }
    return mine < theirs;
  }

};

inline NetStream &operator<<( NetStream &s, const JobID &id)
{
  for( int i=0; i < IDLEN; i++)
    s << (uint8)id.id[i];
  return s;
}

inline NetStream &operator>>( NetStream &s, JobID &id)
{
	for( int i=0; i < IDLEN; i++){
		s.Read( id.id[i] );
		//s >> (uint8) id.id[i];
	}
	return s;
}

inline std::ostream &operator<<( std::ostream &s, JobID &id)
{
  for( int i=0; i < IDLEN; i++)
    s << (uint8) id.id[i];
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
  RESPONSE_FAILED,
  RESPONSE_DATASET,
  //RESPONSE_ERROR_IN_EXECUTION,
  //RESPONSE_ERROR_IN_INPUT
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


/** @} */

