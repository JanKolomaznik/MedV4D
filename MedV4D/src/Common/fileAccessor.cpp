#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/fileAccessor.h"
#include <stdio.h>

using namespace std;
using namespace M4D::IO;

/////////////////////////////////////////////////////////////////////////////
FileAccessor::FileAccessor(const char *file, OpenMode mode) 
{
	Open( file, mode );
}

FileAccessor::FileAccessor(const std::string &file, OpenMode mode)
{
	Open( file, mode );
}

FileAccessor::FileAccessor(const Path &file, OpenMode mode) {
	Open( file.string(), mode );
}

void
FileAccessor::Open(const char *file, OpenMode mode)
{
	stream_.exceptions( std::ios::failbit | std::ios::badbit | std::ios::eofbit );
	switch( mode ) {
	case MODE_READ:
		stream_.open(file, fstream::in | fstream::binary);
		break;		
	case MODE_WRITE:
		stream_.open(file, fstream::out | fstream::binary | fstream::trunc);
		break;
	default:
		ASSERT( false );
	}
	
	if(stream_.fail()) {
		stream_.close();
		_THROW_ ErrorHandling::ExceptionBase( 
			TO_STRING( "Cannot create FileAccessor (mode = " << mode << ") - " << file ) );
	}
}

/////////////////////////////////////////////////////////////////////////////
FileAccessor::~FileAccessor() {
	stream_.flush();
	stream_.close();
}
/////////////////////////////////////////////////////////////////////////////
void 
FileAccessor::PutData(const void *data, size_t length) 
{
	stream_.write((const char*)data, length);
	//stream_.flush();
}
/////////////////////////////////////////////////////////////////////////////
size_t 
FileAccessor::GetData(void *data, size_t length) {
	stream_.read((char*)data, length);
	return stream_.gcount();
}
/////////////////////////////////////////////////////////////////////////////
