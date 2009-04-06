#ifndef CELLREMOTEARRAY_H_
#define CELLREMOTEARRAY_H_

namespace M4D {
namespace Cell {

template<typename T, uint8 BUFSIZE>
class RemoteArrayCell
{
public:
//	RemoteArrayCell(T *array) { SetArray(array); }
//	~RemoteArrayCell() { FlushArray(); }
	
	void push_back(T val)
	{
		m_buf[m_currBuf][m_currPos] = val;
		m_currPos++;
		if(m_currPos == BUFSIZE)
			FlushArray();
	}
	
	void SetArray(T *array)
	{
		m_currFlushedPos = m_arrayBegin = array;
		m_currPos = m_currBuf = 0;
	}
	
	void FlushArray()
	{
		CopyData(m_buf[m_currBuf], m_currFlushedPos, m_currPos);
		m_currFlushedPos += m_currPos;
		m_currBuf = !m_currBuf;
		m_currPos = 0;
	}
	
private:
	void CopyData(T *src, T *dest, size_t size)
	{
		memcpy(dest, src, size * sizeof(T));
	}
	
	T m_buf[2][BUFSIZE];
	bool m_currBuf;
	uint8 m_currPos;
	
	T *m_arrayBegin;
	T *m_currFlushedPos;
};

}
}

#endif /*CELLREMOTEARRAY_H_*/
