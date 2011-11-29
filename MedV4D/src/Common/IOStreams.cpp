
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/IOStreams.h"

namespace M4D
{
namespace IO
{
/////////////////////////////////////////////////////////////////////////////
OutStream::OutStream(MediumAccessor *accessor, bool shared )
	: _accessor(accessor)
{
	Init( accessor );
}

OutStream::OutStream()
	: _accessor(NULL)
{
}

OutStream::~OutStream()
{
	if( !_shared && NULL != _accessor ) {
		delete _accessor;
	}
}

void
OutStream::Init( MediumAccessor *accessor, bool shared  )
{
	if( accessor == NULL ) {
		_THROW_ EBadParameter( "NULL pointer" );
	}
	_accessor = accessor;
	uint8 endian = (uint8) GetEndianess();
	_accessor->PutData( (const void *) &endian, sizeof(uint8));
}

/////////////////////////////////////////////////////////////////////////////
void
OutStream::PutDataBuf( const DataBuffs &bufs)
{
	for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
	{
		_accessor->PutData(it->data, it->len);
	}
}
/////////////////////////////////////////////////////////////////////////////
void
OutStream::PutDataBuf( const DataBuff &buf)
{
	_accessor->PutData(buf.data, buf.len);
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
InStream::InStream(MediumAccessor *accessor, bool shared )
	: _accessor(accessor)
{
	Init( accessor, shared );
}

InStream::InStream()
	: _accessor(NULL), _shared( false )
{

}

InStream::~InStream()
{
	if( !_shared && NULL != _accessor ) {
		delete _accessor;
	}
}

void
InStream::Init( MediumAccessor *accessor, bool shared )
{
	if( accessor == NULL ) {
		_THROW_ EBadParameter( "NULL pointer" );
	}
	_accessor = accessor;

	uint8 e;
	_accessor->GetData( &e, sizeof(uint8));
	Endianness endianess = (Endianness) e;
		
	// if stream's endian is different from curr machine .. swap bytes
	if( endianess != GetEndianess() ) {
		_needSwapBytes = 1;
	} else {
		_needSwapBytes = 0;
	}
	
}
///////////////////////////////////////////////////////////////////////////////
//template< typename T>
//void
//InStream::GetDataBuf( DataBuffs &bufs)
//{
//	
//}
///////////////////////////////////////////////////////////////////////////////
//template< typename T>
//void
//InStream::GetDataBuf( DataBuff &buf)
//{
//	
//}
//
///////////////////////////////////////////////////////////////////////////////
//template< typename T>
//void
//InStream::SwapDataBuf( DataBuff &buf)
//{
//	
//}
/////////////////////////////////////////////////////////////////////////////



}//namespace IO
}//namespace M4D
