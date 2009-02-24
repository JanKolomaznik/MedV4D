#ifndef IOSTREAMS_H_
#define IOSTREAMS_H_

#include <vector>
#include "mediumAccessor.h"

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

class OutStream
{
public:
  OutStream(MediumAccessor *accessor);
  // Serialization support for DataSetSerializators
  void PutDataBuf( const DataBuffs &bufs);
  void PutDataBuf( const DataBuff &buf);
  
  template< typename T>
  void Put(const T what)
  {
  	accessor_->PutData( (const void *)&what, sizeof(T));
  }
  
private:
	MediumAccessor *accessor_;
};

class InStream
{
public:
	InStream(MediumAccessor *accessor);
	
	void GetDataBuf( DataBuffs &bufs);
	void GetDataBuf( DataBuff &buf);
	  
	  template< typename T>
	  void Get( T &what)
	  {
		  accessor_->GetData( (void *)&what, sizeof(T));
	  	if(needSwapBytes_)
	  			SwapBytes<T>(what);
	  }
	  
private:
	uint8 needSwapBytes_;
	MediumAccessor *accessor_;
};

}
}

#endif /*IACCESSSTREAM_H_*/
