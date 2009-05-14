#ifndef CELLREMOTEARRAY_H_
#define CELLREMOTEARRAY_H_

#define REMOTEARRAY_BUF_SIZE 8

namespace M4D {
namespace Cell {

template<typename T, uint8 BUFSIZE>
class RemoteArrayCell
{
public:
	typedef T TRemoteArrayCellBuf[2][REMOTEARRAY_BUF_SIZE];
	
	RemoteArrayCell(TRemoteArrayCellBuf &buffer);
	//RemoteArrayCell(Address array);
	~RemoteArrayCell();
	
	void push_back(T val);
	
	
	
	void SetBeginEnd(T *begin, T *end);
	
	void SetArray(Address array);
	void FlushArray();
	
private:
	//void CopyData(T *src, T *dest, size_t size);
	
	TRemoteArrayCellBuf &m_buf;
	bool m_currBuf;
	uint8 m_currPos;
	
	Address m_arrayBegin;
	Address m_currFlushedPos;
};

template<typename T, uint8 BUFSIZE>
class GETRemoteArrayCell
{
public:
	typedef T TRemoteArrayCellBuf[2][REMOTEARRAY_BUF_SIZE];	
	typedef GETRemoteArrayCell<T, BUFSIZE> Self;
	
	GETRemoteArrayCell(TRemoteArrayCellBuf &buffer);//, T *end);
	//~GETRemoteArrayCell();
	
	void SetArray(Address begin);
	
	T GetCurrVal();
	Self &operator++();
	//bool HasNext() { return (m_currPos != m_arrayEnd); }
	
private:
	void LoadNextPiece();
	void CopyData(T *src, T *dest, size_t size);
	
	TRemoteArrayCellBuf &m_buf;
	bool m_currBuf;
	uint8 m_currPos;
	
	Address m_arrayBegin;
	//T *m_arrayEnd;
	Address m_currLoadedPos;
};

//// this array has to notify PPU when flushed to process it
//template<typename T, uint8 BUFSIZE>
//class PUTRemoteArrayCell
//{
//public:
//	PUTRemoteArrayCell();
//	
//	void push_back(T val);
//	bool IsFull() { return m_currPos==BUFSIZE; }
//	void FlushArray(Address whereToFLush);
//	
//private:
//	void CopyData(T *src, T *dest, size_t size);
//	
//	T m_buf[2][BUFSIZE] __attribute__ ((aligned (128)));
//	bool m_currBuf;
//	uint8 m_currPos;
//	
//	Address m_arrayBegin;
//	Address m_currFlushedPos;
//	
//	uint32 id;		// identification (of layer)
//};

}
}

//include implementation
#include "src/cellRemoteArray.tcc"

#endif /*CELLREMOTEARRAY_H_*/
