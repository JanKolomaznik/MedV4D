#ifndef FILE_STREAMS_H
#define FILE_STREAMS_H

#include "MedV4D/Common/IOStreams.h"
#include "MedV4D/Common/fileAccessor.h"
#include <string>
#include <memory>

namespace M4D
{
namespace IO
{


class FOutStream: public OutStream
{
public:
	FOutStream(const char *file): OutStream(  )
	{
		FileAccessor::Ptr acc;
		try {
			acc = std::make_shared< FileAccessor>( file, MODE_WRITE );
			Init( acc );
		} catch (...) {
			_THROW_ EFileProblem( file );
		}
	}

	FOutStream(std::string file): OutStream(  )
	{
		FileAccessor::Ptr acc;
		try {
			acc = std::make_shared< FileAccessor>( file, MODE_WRITE );
			Init( acc );
		} catch (...) {
			_THROW_ EFileProblem( file );
		}
	}

	~FOutStream()
	{ }
protected:
};

class FInStream: public InStream
{
public:
	FInStream(const char *file): InStream( )
	{
		FileAccessor::Ptr acc;
		try {
			acc = std::make_shared< FileAccessor>( file, MODE_READ );
			Init( acc );
		} catch (...) {
			_THROW_ EFileProblem( file );
		}
	}

	FInStream(std::string file): InStream( )
	{
		FileAccessor::Ptr acc;
		try {
			acc = std::make_shared< FileAccessor>( file, MODE_READ );
			Init( acc );
		} catch (...) {
			_THROW_ EFileProblem( file );
		}
	}

	~FInStream()
	{  }
protected:
};

}
}


#endif /*FILE_STREAMS_H*/
