#ifndef IOSTREAMS_H_
#define IOSTREAMS_H_

#include "Common.h"
#include <vector>
#include "mediumAccessor.h"

namespace M4D
{
namespace IO
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

typedef std::vector< DataBuff > DataBuffs;

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
	OutStream(MediumAccessor *accessor, bool shared = true);

	virtual ~OutStream();
	// Serialization support for DataSetSerializators
	void PutDataBuf( const DataBuffs &bufs);
	void PutDataBuf( const DataBuff &buf);

	template< typename T>
	void Put(const T what)
	{
		_accessor->PutData( (const void *)&what, sizeof(T));
	}
  
protected:
  	OutStream();

	void
	Init( MediumAccessor *accessor, bool shared = true );

	MediumAccessor 	*_accessor;
	bool		_shared;
};

class InStream
{
public:
	InStream(MediumAccessor *accessor, bool shared = true);

	virtual ~InStream();

	template< typename T>
	void GetDataBuf( DataBuffs &bufs)
	{
		for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
		{
			_accessor->GetData(it->data, it->len);
			if( _needSwapBytes )
			{
				SwapDataBuf<T>(*it);
			}
		}
	}
	
	template< typename T>
	void GetDataBuf( DataBuff &buf)
	{
		_accessor->GetData(buf.data, buf.len);
			
		if( _needSwapBytes )
		{
			SwapDataBuf<T>(buf);
		}
	}

	template< typename T>
	void Get( T &what)
	{
		_accessor->GetData( (void *)&what, sizeof(T));
		if( _needSwapBytes ) {
			SwapBytes<T>( what );
		}
	}

	template< typename T>
	void Get( T *what, size_t aCount )
	{
		ASSERT( what != NULL && aCount > 0 );

		DataBuff buffer = DataBuff( static_cast< void * >( what ), aCount * sizeof( T ) );

		GetDataBuf< T >( buffer );
	}
	
	bool
	eof()
	{ return _accessor->eof(); }
protected:
	InStream();
	
	void
	Init( MediumAccessor *accessor, bool shared = true );

	template< typename T>
	void SwapDataBuf( DataBuff &buf)
	{
		register uint32 bufSize = buf.len / sizeof(T);
		T *data = (T *) buf.data;
		for(register uint32 i=0; i<bufSize; i++)
			SwapBytes<T>(data[i]);
	}
	
	uint8 		_needSwapBytes;
	MediumAccessor 	*_accessor;
	bool		_shared;
};


}
}

#endif /*IACCESSSTREAM_H_*/
