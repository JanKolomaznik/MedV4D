
#include "Common.h"
#include "IOStreams.h"

using namespace M4D::IO;

/////////////////////////////////////////////////////////////////////////////
OutStream::OutStream(MediumAccessor *accessor)
	: accessor_(accessor)
{
	uint8 endian = (uint8) GetEndianess();
	accessor_->PutData( (const void *) &endian, sizeof(uint8));
}
/////////////////////////////////////////////////////////////////////////////
void
OutStream::PutDataBuf( const DataBuffs &bufs)
{
	for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
	{
		accessor_->PutData(it->data, it->len);
	}
}
/////////////////////////////////////////////////////////////////////////////
void
OutStream::PutDataBuf( const DataBuff &buf)
{
	accessor_->PutData(buf.data, buf.len);
}
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
InStream::InStream(MediumAccessor *accessor)
	: accessor_(accessor)
{
	uint8 e;
	accessor_->GetData( &e, sizeof(uint8));
	Endianness endianess = (Endianness) e;
		
	// if stream's endian is different from curr machine .. swap bytes
	if(endianess != GetEndianess())
		needSwapBytes_ = 1;
	else
		needSwapBytes_ = 0;
}
/////////////////////////////////////////////////////////////////////////////
void
InStream::GetDataBuf( DataBuffs &bufs)
{
	for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
	{
		accessor_->GetData(it->data, it->len);
	}
}
/////////////////////////////////////////////////////////////////////////////
void
InStream::GetDataBuf( DataBuff &buf)
{
	accessor_->GetData(buf.data, buf.len);
}

/////////////////////////////////////////////////////////////////////////////
