#include "Common.h"
#include "fileStream.h"
#include <stdio.h>


using namespace M4D::IO;
using namespace M4D::Imaging;
using namespace std;

/////////////////////////////////////////////////////////////////////////////
FileStream::FileStream(const char *file, OpenMode mode) {
	if(mode == MODE_READ)
	{
		stream_.open(file, 
			fstream::in | fstream::out | fstream::binary);
		
		uint8 e;
		stream_ >> e;		
		endianess_ = (Endianness) e;
		
		// if stream's endian is different from curr machine .. swap bytes
		if(endianess_ != GetEndianess())
			needSwapBytes_ = 1;
		else
			needSwapBytes_ = 0;
	}
	else {
		stream_.open(file, 
			fstream::in | fstream::out | fstream::binary | fstream::trunc);
	}
}
/////////////////////////////////////////////////////////////////////////////
FileStream::~FileStream() {
	stream_.close();
}
/////////////////////////////////////////////////////////////////////////////
void 
FileStream::PutDataBuf(const DataBuffs &bufs) {
	for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
	{
		stream_.write((const char*)it->data, it->len);
	}
}
/////////////////////////////////////////////////////////////////////////////
void 
FileStream::PutDataBuf(const DataBuff &buf) {
	stream_.write((const char*)buf.data, buf.len);
}
/////////////////////////////////////////////////////////////////////////////
void 
FileStream::GetDataBuf(DataBuffs &bufs) {
	for( DataBuffs::const_iterator it=bufs.begin(); it != bufs.end(); it++)
	{
		stream_.read((char*)it->data, it->len);
	}
}
/////////////////////////////////////////////////////////////////////////////
void 
FileStream::GetDataBuf(DataBuff &buf) {
	stream_.read((char*)buf.data, buf.len);
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator<< (const uint8 what)
{
	stream_.write((const char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator<< (const uint16 what)
{
	stream_.write((const char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator<< (const uint32 what)
{
	stream_.write((const char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator<< (const uint64 what)
{
	stream_.write((const char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator<< (const float32 what)
{
	stream_.write((const char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator<< (const float64 what)
{
	stream_.write((const char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator>>( uint8 &what)
{
	stream_.read((char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator>>( uint16 &what)
{
	stream_.read((char*)&what, sizeof(what));
	if(needSwapBytes_)
		SwapBytes(what);
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator>>( uint32 &what)
{
	stream_.read((char*)&what, sizeof(what));
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator>>( uint64 &what)
{
	stream_.read((char*)&what, sizeof(what));
	if(needSwapBytes_)
			SwapBytes(what);
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator>>( float32 &what)
{
	stream_.read((char*)&what, sizeof(what));
	if(needSwapBytes_)
			SwapBytes((uint32 &)what);
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
FileStream& FileStream::operator>>( float64 &what)
{
	stream_.read((char*)&what, sizeof(what));
	if(needSwapBytes_)
			SwapBytes((uint64 &)what);
	return *this;
}
/////////////////////////////////////////////////////////////////////////////
