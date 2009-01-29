#ifndef FILESTREAM_H_
#define FILESTREAM_H_

#include <fstream>
#include "Imaging/iAccessStream.h"
#include "Endianess.h"

namespace M4D
{
namespace IO
{

enum OpenMode
{
	MODE_READ,
	MODE_WRITE
};

class FileStream : public M4D::Imaging::iAccessStream
{
public:
	FileStream(const char *file, OpenMode mode);
	~FileStream();
	
  void PutDataBuf( const M4D::Imaging::DataBuffs &bufs);
  void PutDataBuf( const M4D::Imaging::DataBuff &buf);
  
  void GetDataBuf( M4D::Imaging::DataBuffs &bufs);
  void GetDataBuf( M4D::Imaging::DataBuff &buf);
  
  // basic data types operators
  FileStream& operator<< (const uint8 what);
  FileStream& operator<< (const uint16 what);
  FileStream& operator<< (const uint32 what);
  FileStream& operator<< (const uint64 what);
  FileStream& operator<< (const float32 what);
  FileStream& operator<< (const float64 what);

    ////////
  FileStream& operator>>( uint8 &what);
  FileStream& operator>>( uint16 &what);
  FileStream& operator>>( uint32 &what);
  FileStream& operator>>( uint64 &what);
  FileStream& operator>>( float32 &what);
  FileStream& operator>>( float64 &what);
private:
	std::fstream stream_;
	Endianness endianess_;
	uint8 needSwapBytes_;
};

}
}
#endif /*FILESTREAM_H_*/
