#ifndef CELLREMOTEARRAY_H_
#error File cellRemoteArray.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
RemoteArrayCell<T, BUFSIZE>::RemoteArrayCell()
	: m_arrayBegin(0)
{
	
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
RemoteArrayCell<T, BUFSIZE>::RemoteArrayCell(T *array)
{ 
	SetArray(array);
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
RemoteArrayCell<T, BUFSIZE>::~RemoteArrayCell() 
{
	FlushArray(); 
}
///////////////////////////////////////////////////////////////////////////////	
template<typename T, uint8 BUFSIZE>
void
RemoteArrayCell<T, BUFSIZE>::push_back(T val)
{
	m_buf[m_currBuf][m_currPos] = val;
	m_currPos++;
	if(m_currPos == BUFSIZE)
		FlushArray();
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
RemoteArrayCell<T, BUFSIZE>::SetArray(T *array)
	{
		m_currFlushedPos = m_arrayBegin = array;
		m_currPos = m_currBuf = 0;
	}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
RemoteArrayCell<T, BUFSIZE>::FlushArray()
	{
		CopyData(m_buf[m_currBuf], m_currFlushedPos, m_currPos);
		m_currFlushedPos += m_currPos;
		m_currBuf = !m_currBuf;
		m_currPos = 0;
	}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
RemoteArrayCell<T, BUFSIZE>::CopyData(T *src, T *dest, size_t size)
	{
		memcpy(dest, src, size * sizeof(T));
	}
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
GETRemoteArrayCell<T, BUFSIZE>::GETRemoteArrayCell()//, T *end)
	: m_currBuf(0), m_currPos(0), m_arrayBegin(0), m_currLoadedPos(0)//, m_arrayEnd(end)
{
	
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
GETRemoteArrayCell<T, BUFSIZE>::SetArray(T *begin)
{
	m_currBuf = 0;
	m_currPos = 0;
	m_arrayBegin = m_currLoadedPos = begin;
	
	// load the first chunk
	LoadNextPiece();
	m_currBuf = ! m_currBuf;
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
GETRemoteArrayCell<T, BUFSIZE>::CopyData(T *src, T *dest, size_t size)
{
	memcpy(dest, src, size * sizeof(T));
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
T
GETRemoteArrayCell<T, BUFSIZE>::GetCurrVal()
{
	return m_buf[m_currBuf][m_currPos];
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
GETRemoteArrayCell<T, BUFSIZE> &
GETRemoteArrayCell<T, BUFSIZE>::operator++()
{
	m_currPos++;
	if(m_currPos == BUFSIZE)
	{
		LoadNextPiece();
		m_currBuf = ! m_currBuf;
		m_currPos = 0;
	}
	return *this;
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
GETRemoteArrayCell<T, BUFSIZE>::LoadNextPiece()
{
	CopyData(m_currLoadedPos, m_buf[!m_currBuf], BUFSIZE);
	m_currLoadedPos += BUFSIZE;
}
///////////////////////////////////////////////////////////////////////////////

}}

#endif
