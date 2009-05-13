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

	inline bool HasNext(void) { 
		return (GetCurrItem()->Next != m_end); 
		}	

	Item *Next(void);	
	Address GetCentralMemAddrrOfCurrProcessedNode() { return m_realAddresses[m_currBufPosition]; }
	
private:
	
	Item * GetCurrItem() { return &m_buf[m_currBufPosition]; }
	
//	inline bool HasNextForLoad(void) { return (GetCurrItem()->Next != m_end); }
	//void Load(Address src, Item *dest, size_t size);
	
	Item m_buf[2];
	Address m_realAddresses[2];
	
	uint8 m_currBufPosition;
	
	//Item *m_begin;
	Address m_end;
	//Item *m_nextForLoad;
	//Item *m_currToProcess;
	
	//Item *m_currProcessedCentralMemAddress;
	
	Address pom;
	uint32 counter;
	
	unsigned int tag;
};

//include implementation
#include "src/cellLinkedChainIterator.tcc"

}}  // namespace



#endif /*LINKEDCHAINITERATOR_H_*/
