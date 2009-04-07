#ifndef FILE_STREAMS_H
#define FILE_STREAMS_H

#include "common/IOStreams.h"
#include "common/fileAccessor.h"
#include <string>

namespace M4D
{
namespace IO
{


class FOutStream: public OutStream
{
public:
	FOutStream(const char *file): OutStream( new FileAccessor( file, MODE_WRITE ) )
	{ _fAccessor = accessor_; }

	FOutStream(std::string file): OutStream( new FileAccessor( file.data(), MODE_WRITE ) )
	{ _fAccessor = accessor_; }

	~FOutStream()
	{ delete _fAccessor; }
protected:
	MediumAccessor *_fAccessor;
};

class FInStream: public InStream
{
public:
	FInStream(const char *file): InStream( new FileAccessor( file, MODE_READ ) )
	{ _fAccessor = accessor_; }

	FInStream(std::string file): InStream( new FileAccessor( file.data(), MODE_READ ) )
	{ _fAccessor = accessor_; }

	~FInStream()
	{ delete _fAccessor; }
protected:
	MediumAccessor *_fAccessor;
};

}
}


#endif /*FILE_STREAMS_H*/
