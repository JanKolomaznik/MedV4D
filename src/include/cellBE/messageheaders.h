#ifndef MESSAGE_HEADERS_H
#define MESSAGE_HEADERS_H

#include "netStream.h"

namespace M4D
{
namespace CellBE
{
  
#define IDLEN 12
typedef uint8* JobID;

struct PrimaryJobHeader
{
  uint8 id[IDLEN];
  
  uint8 action;

  uint8 dataSetType;

  uint8 endian;  
  
  static void Serialize( PrimaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( PrimaryJobHeader) );

    for( int i=0; i < IDLEN; i++)
      s << h->id[i];
    s << h->action << h->dataSetType;
#ifdef LITTLE_ENDIAN
    s << (uint8) 0;
#else
    s << (uint8) 1;
#endif
  }

  static void Deserialize( PrimaryJobHeader *h)
  {
    NetStreamArrayBuf s( (uint8 *)h, sizeof( PrimaryJobHeader) );

    for( int i=0; i < IDLEN; i++)
      s >> h->id[i];
    s >> h->action;
  }
};

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