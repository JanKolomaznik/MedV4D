#include "common/Common.h"
#include "common/fileAccessor.h"
#include <stdio.h>

using namespace std;
using namespace M4D::IO;

/////////////////////////////////////////////////////////////////////////////
FileAccessor::FileAccessor(const char *file, OpenMode mode) {
	if(mode == MODE_READ)
	{
		stream_.open(file, 
			fstream::in | fstream::binary);
		if(stream_.fail())
		{
			stream_.close();
			throw exception();
		}
	}
	else {
		stream_.open(file, 
			fstream::out | fstream::binary | fstream::trunc);
		if(stream_.fail())
		{
			stream_.close();
			throw exception();
		}
	}
}
/////////////////////////////////////////////////////////////////////////////
FileAccessor::~FileAccessor() {
	stream_.close();
}
/////////////////////////////////////////////////////////////////////////////
void 
FileAccessor::PutData(const void *data, size_t length) 
{
	stream_.write((const char*)data, length);
}
/////////////////////////////////////////////////////////////////////////////
void 
FileAccessor::GetData(void *data, size_t length) {
	stream_.read((char*)data, length);
}
/////////////////////////////////////////////////////////////////////////////
