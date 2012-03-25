#ifndef IOSTREAMS_H_
#define IOSTREAMS_H_

#include "MedV4D/Common/Common.h"
#include <vector>
#include "MedV4D/Common/mediumAccessor.h"

namespace M4D
{
namespace IO
{

struct DataBuff
{
  DataBuff( void *dat, size_t lenght)
    : data( dat)
    , len(lenght)
    , filled( 0 )
  {}

  DataBuff() {}

  DataBuff( const DataBuff &b)  // copy ctor
    : data( b.data)
    , len( b.len)
  {}

  void *data;
  size_t len;
  size_t filled;
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
	OutStream( MediumAccessor::Ptr accessor );

	virtual ~OutStream();
	// Serialization support for DataSetSerializators
	void PutDataBuf( const DataBuffs &bufs);
	void PutDataBuf( const DataBuff &buf);

	template< typename T>
	void Put(const T what)
	{
		_accessor->PutData( (const void *)&what, sizeof(T));
		
		mProcessedBytes += sizeof(T);
	}
  
	template< typename T>
	void Put( const T *what, size_t aCount )
	{
		ASSERT( what != NULL && aCount > 0 );

		DataBuff buffer = DataBuff( static_cast< void *>( const_cast< T * >( what ) ), aCount * sizeof( T ) );

		PutDataBuf( buffer );
	}
protected:
  	OutStream();

	void
	Init( MediumAccessor::Ptr accessor );

	MediumAccessor::Ptr _accessor;
	bool		_shared;
	size_t		mProcessedBytes;
};

class InStream
{
public:
	InStream( MediumAccessor::Ptr accessor );

	virtual ~InStream();

	template< typename T>
	void GetDataBuf( DataBuffs &bufs)
	{
		for( DataBuffs::iterator it=bufs.begin(); it != bufs.end(); it++)
		{
			it->filled = _accessor->GetData(it->data, it->len);
			mProcessedBytes += it->filled;
			if( _needSwapBytes )
			{
				SwapDataBuf<T>(*it);
			}
		}
	}
	
	template< typename T>
	void GetDataBuf( DataBuff &buf)
	{
		buf.filled = _accessor->GetData(buf.data, buf.len);
		mProcessedBytes += buf.filled;
		//TODO handle not if not read whole
		if( _needSwapBytes )
		{
			SwapDataBuf<T>(buf);
		}
	}

	template< typename T>
	void Get( T &what)
	{
		size_t bytes = _accessor->GetData( (void *)&what, sizeof(T));
		mProcessedBytes +=  bytes;
		if ( bytes != sizeof(T) ) {
			_THROW_ ErrorHandling::ExceptionBase( TO_STRING( "Stream single data read failed. Data already processed: " << mProcessedBytes << " bytes" ) );
		}
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
		if ( buffer.filled < aCount * sizeof( T ) ) {
			_THROW_ ErrorHandling::ExceptionBase( TO_STRING( "Stream multiple data read failed: wanted " << (aCount * sizeof( T )) 
						<< " bytes; read " << buffer.filled << " bytes. Data already processed: " << mProcessedBytes << " bytes" ) );
		}
	}
	
	bool
	eof()
	{ return _accessor->eof(); }
protected:
	InStream();
	
	void
	Init( MediumAccessor::Ptr accessor );

	template< typename T>
	void SwapDataBuf( DataBuff &buf)
	{
		register size_t bufSize = buf.len / sizeof(T);
		T *data = (T *) buf.data;
		for(register size_t i=0; i<bufSize; i++)
			SwapBytes<T>(data[i]);
	}
	
	uint8 		_needSwapBytes;
	MediumAccessor::Ptr _accessor;
	bool		_shared;
	size_t		mProcessedBytes;
};


}
}

#endif /*IACCESSSTREAM_H_*/
