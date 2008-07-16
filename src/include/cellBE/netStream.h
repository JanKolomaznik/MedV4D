#ifndef NETSTREAM_H
#define NETSTREAM_H

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
  virtual NetStream & operator<< (const uint16 what) = 0;
  virtual NetStream & operator<< (const uint32 what) = 0;
  virtual NetStream & operator<< (const float32 what) = 0;
  virtual NetStream & operator>>( uint8 &what) = 0;
  virtual NetStream & operator>>( uint16 &what) = 0;
  virtual NetStream & operator>>( uint32 &what) = 0;
  virtual NetStream & operator>>( float32 &what) = 0;
};

}
}

#endif
