/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file netStream.h 
 * @{ 
 **/

#ifndef NETSTREAM_H
#define NETSTREAM_H

#include "Common.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Interface that is given to user of Imaging library. Through virtual
 *  function is actual implementation hidden in CellBE library.
 */
class NetStream
{

public:
  virtual NetStream & operator<< (const uint8 what) = 0;
  NetStream & operator<< (const int8 what)
  {
    return operator<<( (const uint8) what);
  }

  virtual NetStream & operator<< (const uint16 what) = 0;
  NetStream & operator<< (const int16 what)
  {
    return operator<<( (const uint16) what);
  }

  virtual NetStream & operator<< (const uint32 what) = 0;
  NetStream & operator<< (const int32 what)
  {
    return operator<<( (const uint32) what);
  }

  virtual NetStream & operator<< (const uint64 what) = 0;
  NetStream & operator<< (const int64 what)
  {
    return operator<<( (const uint64) what);
  }

  virtual NetStream & operator<< (const float32 what) = 0;
  virtual NetStream & operator<< (const float64 what) = 0;

  ////////
  virtual NetStream & operator>>( uint8 &what) = 0;
  NetStream & operator>>( int8 &what)
  {
    return operator>>( (uint8 &) what);
  }

  virtual NetStream & operator>>( uint16 &what) = 0;
  NetStream & operator>>( int16 &what)
  {
    return operator>>( (uint16 &) what);
  }

  virtual NetStream & operator>>( uint32 &what) = 0;
  NetStream & operator>>( int32 &what)
  {
    return operator>>( (uint32 &) what);
  }

  virtual NetStream & operator>>( uint64 &what) = 0;
  NetStream & operator>>( int64 &what)
  {
    return operator>>( (uint64 &) what);
  }

  virtual NetStream & operator>>( float32 &what) = 0;
  virtual NetStream & operator>>( float64 &what) = 0;
};

}
}

#endif

/** @} */

