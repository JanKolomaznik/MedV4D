#ifndef IACCESSSTREAM_H_
#define IACCESSSTREAM_H_

#include <vector>
namespace M4D
{
namespace Imaging
{

struct DataBuff
{
  DataBuff( void *dat, size_t lenght)
    : data( dat)
    , len(lenght)
  {}

  DataBuff() {}

  DataBuff( const DataBuff &b)  // copy ctor
    : data( b.data)
    , len( b.len)
  {}

  void *data;
  size_t len;
};

class DataBuffs : public std::vector< DataBuff >
{
};

/**
 * Interface that provides access to dataSet. It can use network
 * or file to get data from.
 * Larger chunks of data are handled through DataBuffs and are not
 * affected by endianness. Endian issues shall be handled along with
 * the data.
 * Another functions are operators that handle basic data types and
 * also handle the endian ussies so returned data are already in
 * right endian.
 */

class iAccessStream
{
public:

  virtual ~iAccessStream();

  // Serialization support for DataSetSerializators
  virtual void PutDataBuf( const DataBuffs &bufs) = 0;
  virtual void PutDataBuf( const DataBuff &buf) = 0;
  
  virtual void GetDataBuf( DataBuffs &bufs) = 0;
  virtual void GetDataBuf( DataBuff &buf) = 0;
  
  // basic data types operators
  virtual iAccessStream & operator<< (const uint8 what) = 0;
	iAccessStream & operator<< (const int8 what)
	{
	  return operator<<( (const uint8) what);
	}

    virtual iAccessStream & operator<< (const uint16 what) = 0;
    iAccessStream & operator<< (const int16 what)
    {
      return operator<<( (const uint16) what);
    }

    virtual iAccessStream & operator<< (const uint32 what) = 0;
    iAccessStream & operator<< (const int32 what)
    {
      return operator<<( (const uint32) what);
    }

    virtual iAccessStream & operator<< (const uint64 what) = 0;
    iAccessStream & operator<< (const int64 what)
    {
      return operator<<( (const uint64) what);
    }

    virtual iAccessStream & operator<< (const float32 what) = 0;
    virtual iAccessStream & operator<< (const float64 what) = 0;

    ////////
    virtual iAccessStream & operator>>( uint8 &what) = 0;
    iAccessStream & operator>>( int8 &what)
    {
      return operator>>( (uint8 &) what);
    }

    virtual iAccessStream & operator>>( uint16 &what) = 0;
    iAccessStream & operator>>( int16 &what)
    {
      return operator>>( (uint16 &) what);
    }

    virtual iAccessStream & operator>>( uint32 &what) = 0;
    iAccessStream & operator>>( int32 &what)
    {
      return operator>>( (uint32 &) what);
    }

    virtual iAccessStream & operator>>( uint64 &what) = 0;
    iAccessStream & operator>>( int64 &what)
    {
      return operator>>( (uint64 &) what);
    }

    virtual iAccessStream & operator>>( float32 &what) = 0;
    virtual iAccessStream & operator>>( float64 &what) = 0;

};

}
}

#endif /*IACCESSSTREAM_H_*/
