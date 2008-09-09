/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file netStreamImpl.h 
 * @{ 
 **/

#ifndef NETSTREAM_IMPL_H
#define NETSTREAM_IMPL_H

#ifdef __unix__
#include <arpa/inet.h>
#endif

#include <vector>

#include "cellBE/netStream.h"

namespace M4D
{
namespace CellBE
{

  /**
   *  Actual implementation of NetStream interface. In this class are 
   *  serializing and deserializing operators of some common data types
   *  implemented. Also contains pure virtual functions for byte 
   *  putting and retrival alowing different implementation in successors.
   */
class NetStreamBaseImpl : public NetStream
{
protected:
  void virtual AddByte( uint8) = 0;
  bool virtual HasNext( void) = 0;    // tels true if there are more data
  uint8 virtual GetByte(void) = 0;

  uint16 supp16;
  uint32 supp32;
  uint64 supp64;
  uint8 *ptr8;

public:
  NetStream & operator<< (const uint8 what)
  {
    AddByte( what);
    return *this;
  }

  NetStream & operator<< (const uint16 what)
  {
    supp16 = htons(what);  // convert it to network representation

    AddByte( ((uint8*)&supp16)[0] );
    AddByte( ((uint8*)&supp16)[1] );
    return *this;
  }

  NetStream & operator<< (const uint32 what)
  {
    supp32 = htonl(what);  // convert it to network representation

    AddByte( ((uint8*)&supp32)[0] );
    AddByte( ((uint8*)&supp32)[1] );
    AddByte( ((uint8*)&supp32)[2] );
    AddByte( ((uint8*)&supp32)[3] );
    return *this;
  }

  NetStream & operator<< (const uint64 what)
  {
    supp64 = *((uint64*) &what);
    uint8 mask = 255; // full byte

    // FOR REVISOIN
    for( uint8 i=7; i >= 0; i--)
    {
      AddByte( (uint8) ( (supp64 | (mask << (i*8))) >> (i*8) ) );
    }
    return *this;
  }

  NetStream & operator<< (const float32 what)
  {
    supp32 = htonl(*((uint32*)&what));  // convert it to network representation

    AddByte( ((uint8*)&supp32)[0] );
    AddByte( ((uint8*)&supp32)[1] );
    AddByte( ((uint8*)&supp32)[2] );
    AddByte( ((uint8*)&supp32)[3] );
    return *this;
  }

  NetStream & operator<< (const float64 what)
  {
    supp64 = *((uint64*) &what);
    return operator<<(supp64);
  }

  NetStream & operator>>( uint8 &what)
  {
    what = GetByte();
    return *this;
  }

  NetStream & operator>>( uint16 &what)
  {
    ptr8 = (uint8*)&supp16;

    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    what = ntohs( supp16);

    return *this;
  }

  NetStream & operator>>( uint32 &what)
  {
    ptr8 = (uint8*)&supp32;
    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    ptr8[2] = GetByte();
    ptr8[3] = GetByte();

    what = ntohl( supp32);
    
    return *this;
  }

  NetStream & operator>>( uint64 &what)
  {
    // FOR REVIZION
    ptr8 = (uint8*)&supp64;

    if( GetEndianess() == End_BIG_ENDIAN)
    {
      ptr8[0] = GetByte();
      ptr8[1] = GetByte();
      ptr8[2] = GetByte();
      ptr8[3] = GetByte();
      ptr8[4] = GetByte();
      ptr8[5] = GetByte();
      ptr8[6] = GetByte();
      ptr8[7] = GetByte();
    }
    else
    {
      ptr8[7] = GetByte();    // just in swapped order
      ptr8[6] = GetByte();
      ptr8[5] = GetByte();
      ptr8[4] = GetByte();
      ptr8[3] = GetByte();
      ptr8[2] = GetByte();
      ptr8[1] = GetByte();
      ptr8[0] = GetByte();
    }

    what = supp64;
    
    return *this;
  }

  NetStream & operator>>( float32 &what)
  {
    ptr8 = (uint8*)&supp32;
    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    ptr8[2] = GetByte();
    ptr8[3] = GetByte();

    supp32 = ntohl( *((uint32*)&supp32));
    what = *( (float32*)&supp32);

    return *this;
  }

  NetStream & operator>>( float64 &what)
  {
    // FOR REVIZION
    return operator>>( (uint64 &)what);
  }
};

///////////////////////////////////////////////////////////////////////////////

/**
 *  NetStream based on memory buffer. It manages underlaying buffer
 *  interpreted as an array of bytes.
 */
class NetStreamArrayBuf : public NetStreamBaseImpl
{
  uint8 *begin; // underlaying buf
  uint8 *curr;
  uint8 *end;

public:

  bool HasNext( void)
  {
    return curr != end;
  }

  NetStreamArrayBuf( uint8 *buf, size_t size)
  {
    begin = buf;
    curr = buf;
    end = buf + size; 
  }

  void AddByte( uint8 what)
  {
    if( curr != end)
    {
      *curr = what;
      curr++;
    }
  }

  uint8 GetByte( void)
  {
    if( curr != end)
      return *curr++;
    else
      throw ExceptionBase("Already on end of stream");
  }

};

///////////////////////////////////////////////////////////////////////////////

/**
 *  NetStream based on underlaying std::vector of bytes.
 */
class NetStreamVector 
  : public std::vector<uint8>, public NetStreamBaseImpl
{
  size_t curr;
public:

  bool HasNext( void)
  {
    return curr < size();
  }

  NetStreamVector() : curr(0) {}

  void AddByte( uint8 what)
  {
    push_back( what);
  }

  uint8 GetByte( void)
  {
    if( curr < size())
      return this->at(curr++);
    else
      throw ExceptionBase("Already on end of stream");
  }

};

///////////////////////////////////////////////////////////////////////////////

/**
 *  Base class for Serailizing netstream given to user through iPublicJob
 *  interface to be able to serialize some structured data. It serialize
 *  data in endian of the mashine thet performs the serialization (source).
 *  Because that endian goes with the data, right byte ordering is preformed
 *  on destination mashine throuh <<operators in deriving classes.
 */  
class UserSerializingNetStreamBase : public NetStream
{
protected:
  uint8 *begin; // underlaying buf
  uint8 *end;
  uint8 *curr;

  uint16 supp16;
  uint32 supp32;
  uint64 supp64;

  uint8 *ptr8;

  inline void AddByte( uint8 what)
  {
    if( curr != end)
    {
      *curr = what;
      curr++;
    }
  }

  inline uint8 GetByte( void)
  {
    if( curr != end)
      return *curr++;
    else
      throw ExceptionBase("Already on end of stream");
  }  

  UserSerializingNetStreamBase() : begin(NULL), end(NULL), curr(NULL) {}

public:
  inline void SetBuffer( uint8 *buff, size_t size)
  {
    begin = curr = buff;
    end = begin + size;
  }

  NetStream & operator<< (const uint8 what)
  {
    AddByte( what);
    return *this;
  }

  NetStream & operator>>( uint8 &what)
  {
    what = GetByte();
    return *this;
  }

  NetStream & operator<< (const uint16 what)
  {
    supp16 = what;

    AddByte( ((uint8*)&supp16)[0] );
    AddByte( ((uint8*)&supp16)[1] );
    return *this;
  }

  NetStream & operator<< (const uint32 what)
  {
    supp32 = htonl(what);  // convert it to network representation

    AddByte( ((uint8*)&supp32)[0] );
    AddByte( ((uint8*)&supp32)[1] );
    AddByte( ((uint8*)&supp32)[2] );
    AddByte( ((uint8*)&supp32)[3] );
    return *this;
  }

  NetStream & operator<< (const uint64 what)
  {
    supp64 = *((uint64*)&what);

    AddByte( ((uint8*)&supp64)[0] );
    AddByte( ((uint8*)&supp64)[1] );
    AddByte( ((uint8*)&supp64)[2] );
    AddByte( ((uint8*)&supp64)[3] );
    AddByte( ((uint8*)&supp64)[4] );
    AddByte( ((uint8*)&supp64)[5] );
    AddByte( ((uint8*)&supp64)[6] );
    AddByte( ((uint8*)&supp64)[7] );

    return *this;
  }

  NetStream & operator<< (const float32 what)
  {
    supp32 = *((uint32*)&what);

    AddByte( ((uint8*)&supp32)[0] );
    AddByte( ((uint8*)&supp32)[1] );
    AddByte( ((uint8*)&supp32)[2] );
    AddByte( ((uint8*)&supp32)[3] );
    return *this;
  }

  NetStream & operator<< (const float64 what)
  {
    return operator<<( (const uint64) what);
  }

};

///////////////////////////////////////////////////////////////////////////////

/**
 *  Retrieves data and DOES NOT swap them
 */
class UserSerializingNetStreamNotSwapping : public UserSerializingNetStreamBase
{
public:

  NetStream & operator>>( uint16 &what)
  {
    ptr8 = (uint8*)&supp16;

    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    what = supp16;

    return *this;
  }

  NetStream & operator>>( uint32 &what)
  {
    ptr8 = (uint8*)&supp32;
    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    ptr8[2] = GetByte();
    ptr8[3] = GetByte();

    what = supp32;
    
    return *this;
  }

  NetStream & operator>>( uint64 &what)
  {
    // TO BE REVIZED
    ptr8 = (uint8*)&supp64;
    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    ptr8[2] = GetByte();
    ptr8[3] = GetByte();
    ptr8[4] = GetByte();
    ptr8[5] = GetByte();
    ptr8[6] = GetByte();
    ptr8[7] = GetByte();

    what = supp64;
    
    return *this;
  }

  NetStream & operator>>( float32 &what)
  {
    ptr8 = (uint8*)&supp32;
    ptr8[0] = GetByte();
    ptr8[1] = GetByte();
    ptr8[2] = GetByte();
    ptr8[3] = GetByte();

    supp32 = *((uint32*)&supp32);
    what = *( (float32*)&supp32);

    return *this;
  }

  NetStream & operator>>( float64 &what)
  {
    // TO BE REVIZED
    return operator>>( (uint64 &)what);
  }
};

///////////////////////////////////////////////////////////////////////////////

/**
 *  Automatically swaps bytes when retrieving data
 */
class UserSerializingNetStreamSwapping : public UserSerializingNetStreamBase
{
public:

  NetStream & operator>>( uint16 &what)
  {
    ptr8 = (uint8*)&supp16;

    ptr8[1] = GetByte();
    ptr8[0] = GetByte();
    what = supp16;

    return *this;
  }

  NetStream & operator>>( uint32 &what)
  {
    ptr8 = (uint8*)&supp32;
    ptr8[3] = GetByte();
    ptr8[2] = GetByte();
    ptr8[1] = GetByte();
    ptr8[0] = GetByte();

    what = supp32;
    
    return *this;
  }

  NetStream & operator>>( uint64 &what)
  {
    // TO BE REVIZED
    ptr8 = (uint8*)&supp64;

    // same as not waping but vice versa order
    ptr8[7] = GetByte();
    ptr8[6] = GetByte();
    ptr8[5] = GetByte();
    ptr8[4] = GetByte();
    ptr8[3] = GetByte();
    ptr8[2] = GetByte();
    ptr8[1] = GetByte();
    ptr8[0] = GetByte();

    what = supp64;
    
    return *this;
  }

  NetStream & operator>>( float32 &what)
  {
    ptr8 = (uint8*)&supp32;
    ptr8[3] = GetByte();
    ptr8[2] = GetByte();
    ptr8[1] = GetByte();
    ptr8[0] = GetByte();

    supp32 = *((uint32*)&supp32);
    what = *( (float32*)&supp32);

    return *this;
  }

  NetStream & operator>>( float64 &what)
  {
    // TO BE REVIZED
    return operator>>( (uint64 &)what);
  }
};

}
}

#endif

/** @} */

