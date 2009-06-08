#ifndef LINKEDCHAINITERATOR_H_
#define LINKEDCHAINITERATOR_H_

#include "DMAGate.h"

namespace M4D {
namespace Cell {

#define DEBUG_CHAINTOOL 12

template<typename Item>
class LinkedChainIteratorCell
{
public:
	LinkedChainIteratorCell();
	~LinkedChainIteratorCell();
	
	void SetBeginEnd(Address begin, Address end);

	inline bool HasNextToLoad(void) { return m_buf[_loadedPos].Next != m_end; }
	
	bool HasNext()
	{
		return m_buf[m_currBufPosition].Next != m_end;
	}

	Item *GetLoaded(void);
	void Next();
	Address GetCentralMemAddrrOfCurrProcessedNode() { 
		return m_realAddresses[m_currBufPosition]; 
		}
	
	Item * GetCurrItem() { return &m_buf[m_currBufPosition]; }
	
#ifdef FOR_CELL
	void ReserveTag();
	void ReturnTag();
#endif
	
private:
	
	
//	inline bool HasNextForLoad(void) { return (GetCurrItem()->Next != m_end); }
	//void Load(Address src, Item *dest, size_t size);
#define BUF_SIZE 3
	Item m_buf[BUF_SIZE] __attribute__ ((aligned (128)));
	Address m_realAddresses[BUF_SIZE];
	
	uint8 m_currBufPosition;
	uint8 _loadingPos;
	uint8 _loadedPos;
	
	//Item *m_begin;
	Address m_end;
	//Item *m_nextForLoad;
	//Item *m_currToProcess;
	
	//Item *m_currProcessedCentralMemAddress;
	
	Address loadedIter;
	Address processedIter;
	uint32 counter;
	
	int8 _tag;
};

//include implementation
#include "src/cellLinkedChainIterator.tcc"

}}  // namespace



#endif /*LINKEDCHAINITERATOR_H_*/
