#ifndef FILESTREAM_H_
#define FILESTREAM_H_

#include <fstream>
#include "IOStreams.h"

namespace M4D
{
namespace IO
{

enum OpenMode
{
	MODE_READ,
	MODE_WRITE
};

class FileAccessor : public M4D::IO::MediumAccessor
{
public:
	FileAccessor(const char *file, OpenMode mode);
	FileAccessor(const std::string &file, OpenMode mode);
	FileAccessor(const Path &file, OpenMode mode);
	~FileAccessor();
	
	void PutData(const void *data, size_t length);
	size_t GetData(void *data, size_t length);

	bool eof()
	{ return stream_.eof(); }
private:
	void
	Open(const std::string &file, OpenMode mode)
	{
		Open( file.data(), mode );
	}
	void
	Open(const char *file, OpenMode mode);

	std::fstream stream_;
};

}
}
#endif /*FILESTREAM_H_*/
