#ifndef LINKEDCHAINITERATOR_H_
#define LINKEDCHAINITERATOR_H_

#ifdef FOR_CELL
#include <spu_mfcio.h>
#endif

namespace M4D {
namespace Cell {

#define DEBUG_CHAINTOOL 12

template<typename Item>
class LinkedChainIteratorCell
{
public:
	LinkedChainIteratorCell();
	~LinkedChainIteratorCell();
	
	void SetBeginEnd(Item *begin, Item *end);

	inline bool HasNext(void) { 
		return (GetCurrItem()->Next != m_end); 
		}	

	Item *Next(void);	
	Item *GetCentralMemAddrrOfCurrProcessedNode() { return m_realAddresses[m_currBufPosition]; }
	
private:
	
	Item * GetCurrItem() { return &m_buf[m_currBufPosition]; }
	
//	inline bool HasNextForLoad(void) { return (GetCurrItem()->Next != m_end); }
	void Load(Item *src, Item *dest, size_t size);
	
	Item m_buf[2];
	Item *m_realAddresses[2];
	
	uint8 m_currBufPosition;
	
	//Item *m_begin;
	Item *m_end;
	//Item *m_nextForLoad;
	//Item *m_currToProcess;
	
	//Item *m_currProcessedCentralMemAddress;
	
	Item *pom;
	uint32 counter;
	
	unsigned int tag;
};


}}  // namespace

//include implementation
#include "src/cellLinkedChainIterator.tcc"

#endif /*LINKEDCHAINITERATOR_H_*/
