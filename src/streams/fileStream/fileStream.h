#ifndef FILESTREAM_H_
#define FILESTREAM_H_

#include <fstream>
#include "Imaging/IO/IOStreams.h"

namespace M4D
{
namespace IO
{

enum OpenMode
{
	MODE_READ,
	MODE_WRITE
};

class FileAccessor : public M4D::Imaging::MediumAccessor
{
public:
	FileAccessor(const char *file, OpenMode mode);
	~FileAccessor();
	
	void PutData(const void *data, size_t length);
	void GetData(void *data, size_t length);
private:
	std::fstream stream_;
};

}
}
#endif /*FILESTREAM_H_*/
