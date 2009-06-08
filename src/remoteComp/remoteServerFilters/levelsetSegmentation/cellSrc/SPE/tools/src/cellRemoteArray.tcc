#ifndef CELLREMOTEARRAY_H_
#error File cellRemoteArray.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
RemoteArrayCell<T, BUFSIZE>::RemoteArrayCell(TRemoteArrayCellBuf &buffer)
	: m_buf(buffer)
	, m_arrayBegin(0)
{
	
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
RemoteArrayCell<T, BUFSIZE>::SetArray(Address array)
	{
		m_currFlushedPos = m_arrayBegin = array;
		m_currPos = m_currBuf = 0;
#ifdef FOR_CELL
		_tag = DMAGate::GetTag();
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_GET:RemoteArrayCell:%d\n", _tag);
#endif
#endif
	}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
RemoteArrayCell<T, BUFSIZE>::FlushArray()
{
	
//#define ACTIVE_VALS_LEN 224
//#define ACTIVE_VALS_ARR_END (m_arrayBegin.Get64() + (ACTIVE_VALS_LEN * sizeof(T)))
//	if( (m_currFlushedPos.Get64() > ACTIVE_VALS_ARR_END)
//			|| ( (m_currFlushedPos.Get64() + (BUFSIZE * sizeof(T))) > ACTIVE_VALS_ARR_END) )
//			{
//		D_PRINT("PutArrayMimo!: m_currFlushedPos=" << (void*) m_currFlushedPos.Get64()
//				<< "; m_currPos=" << (uint32) m_currPos);
//			}
#ifdef FOR_CELL
	DMAGate::Put(m_buf[m_currBuf], m_currFlushedPos, BUFSIZE * sizeof(T), _tag);
#else
	DMAGate::Put(m_buf[m_currBuf], m_currFlushedPos, BUFSIZE * sizeof(T));
#endif
	m_currFlushedPos += m_currPos * sizeof(T);
	m_currBuf = !m_currBuf;
	m_currPos = 0;
}
///////////////////////////////////////////////////////////////////////////////

#ifdef FOR_CELL
template<typename T, uint8 BUFSIZE>
void
RemoteArrayCell<T, BUFSIZE>::WaitForTransfer()
{
	mfc_write_tag_mask(1 << _tag);
	mfc_read_tag_status_all();
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_RET:RemoteArrayCell:%d\n", _tag);
#endif
	DMAGate::ReturnTag(_tag);
}
#endif

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
GETRemoteArrayCell<T, BUFSIZE>::GETRemoteArrayCell(TRemoteArrayCellBuf &buffer)//, T *end)
	: m_buf(buffer)
	, m_currBuf(0)
	, m_currPos(0)
	, m_arrayBegin(0)
	, m_currLoadedPos(0)
{
	
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint8 BUFSIZE>
void
GETRemoteArrayCell<T, BUFSIZE>::SetArray(Address begin)
{
	m_currBuf = 0;
	m_currPos = 0;
	m_arrayBegin = m_currLoadedPos = begin;

#ifdef FOR_CELL
#ifdef TAG_RETURN_DEBUG
	_tag = DMAGate::GetTag();
	D_PRINT("TAG_GET:GETRemoteArrayCell:%d\n", _tag);
#endif
#endif
	
	// load the first chunk
	LoadNextPiece();
	m_currBuf = ! m_currBuf;
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
#ifdef FOR_CELL
	DMAGate::Get(m_currLoadedPos, m_buf[!m_currBuf], BUFSIZE * sizeof(T), _tag);
#else
	DMAGate::Get(m_currLoadedPos, m_buf[!m_currBuf], BUFSIZE * sizeof(T));
#endif
	m_currLoadedPos += BUFSIZE * sizeof(T);
}
///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
template<typename T, uint8 BUFSIZE>
void
GETRemoteArrayCell<T, BUFSIZE>::WaitForTransfer()
{
	mfc_write_tag_mask(1 << _tag);
	mfc_read_tag_status_all();
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_RET:GETRemoteArrayCell:%d\n", _tag);
#endif
	DMAGate::ReturnTag(_tag);
}
#endif

///////////////////////////////////////////////////////////////////////////////

}}

#endif
