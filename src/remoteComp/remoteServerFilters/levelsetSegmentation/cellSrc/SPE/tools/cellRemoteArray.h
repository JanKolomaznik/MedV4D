#ifndef CELLREMOTEARRAY_H_
#define CELLREMOTEARRAY_H_

namespace M4D {
namespace Cell {

template<typename T, uint8 BUFSIZE>
class RemoteArrayCell
{
public:
	RemoteArrayCell();
	RemoteArrayCell(T *array);
	~RemoteArrayCell();
	
	void push_back(T val);
	void SetArray(T *array);
	void FlushArray();
	
private:
	void CopyData(T *src, T *dest, size_t size);
	
	T m_buf[2][BUFSIZE];
	bool m_currBuf;
	uint8 m_currPos;
	
	T *m_arrayBegin;
	T *m_currFlushedPos;
};

}
}

//include implementation
#include "src/cellRemoteArray.tcc"

#endif /*CELLREMOTEARRAY_H_*/
