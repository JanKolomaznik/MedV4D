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
	FOutStream(const char *file): OutStream(  )
	{ 
		FileAccessor *acc = NULL;
		try {
			acc = new FileAccessor( file, MODE_WRITE );
			Init( acc, false );
		} catch (...) {
			if( NULL == acc ) {
				delete acc;
			}
			_THROW_ EFileProblem( file );
		}
	}

	FOutStream(std::string file): OutStream(  )
	{ 
		FileAccessor *acc = NULL;
		try {
			acc = new FileAccessor( file, MODE_WRITE );
			Init( acc, false );
		} catch (...) {
			if( NULL == acc ) {
				delete acc;
			}
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
		FileAccessor *acc = NULL;
		try {
			acc = new FileAccessor( file, MODE_READ );
			Init( acc, false );
		} catch (...) {
			if( NULL == acc ) {
				delete acc;
			}
			_THROW_ EFileProblem( file );
		}
	}

	FInStream(std::string file): InStream( )
	{
		FileAccessor *acc = NULL;
		try {
			acc = new FileAccessor( file, MODE_READ );
			Init( acc, false );
		} catch (...) {
			if( NULL == acc ) {
				delete acc;
			}
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
