#ifndef CELLREMOTEARRAY_H_
#define CELLREMOTEARRAY_H_

namespace M4D {
namespace Cell {

template<typename T, uint8 BUFSIZE>
class RemoteArrayCell
{
public:
	typedef T TRemoteArrayCellBuf[2][REMOTEARRAY_BUF_SIZE];
	
	RemoteArrayCell(TRemoteArrayCellBuf &buffer);
	~RemoteArrayCell();
	
	void push_back(T val);
	
	void SetBeginEnd(T *begin, T *end);
	
	void SetArray(Address array);
	bool IsFlushNeeded() { return m_currPos != 0; }
	void FlushArray();
	
#ifdef FOR_CELL
	void WaitForTransfer();
#endif
	
private:
	
	TRemoteArrayCellBuf &m_buf;
	bool m_currBuf;
	uint8 m_currPos;
	
	Address m_arrayBegin;
	Address m_currFlushedPos;
	
	int8 _tag;
};

template<typename T, uint8 BUFSIZE>
class GETRemoteArrayCell
{
public:
	typedef T TRemoteArrayCellBuf[2][REMOTEARRAY_BUF_SIZE];	
	typedef GETRemoteArrayCell<T, BUFSIZE> Self;
	
	GETRemoteArrayCell(TRemoteArrayCellBuf &buffer);
	
	void SetArray(Address begin);
	
	T GetCurrVal();
	Self &operator++();
	
#ifdef FOR_CELL
	void WaitForTransfer();
#endif
	
private:
	void LoadNextPiece();
	void CopyData(T *src, T *dest, size_t size);
	
	TRemoteArrayCellBuf &m_buf;
	bool m_currBuf;
	uint8 m_currPos;
	
	Address m_arrayBegin;
	//T *m_arrayEnd;
	Address m_currLoadedPos;
	
	int8 _tag;
};

}
}

//include implementation
#include "src/cellRemoteArray.tcc"

#endif /*CELLREMOTEARRAY_H_*/
