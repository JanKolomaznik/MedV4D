#ifndef LINKEDCHAINITERATOR_H_
#define LINKEDCHAINITERATOR_H_

#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
#include <spu_mfcio.h>
#endif

//#include "../applyUpdateCalc/layerGate.h"

namespace M4D {
namespace Cell {

template<typename Item>
class LinkedChainIteratorCell
{
public:
	LinkedChainIteratorCell();
	~LinkedChainIteratorCell();
	
//	void SetBeginEnd(Item *begin, Item *end);	
	void SetBeginEnd(Item *begin, Item *end)
	{
		m_currToProcess = begin;
		m_end = end;
	}
	inline bool HasNext(void) { return (m_currToProcess != m_end); }	
//	Item *Next(void);
	Item *Next(void) {
		m_currToProcess = m_currToProcess->Next;
		return m_currToProcess->Previous;
	}
	
private:
	
	inline bool HasNextForLoad(void) { return (m_nextForLoad != m_end); }	
	void Load(Item *src, Item *dest, size_t size);
	
	Item m_buf[2];
	uint8 m_currBufPosition;
	
	Item *m_begin, *m_end;
	Item *m_nextForLoad;
	Item *m_currToProcess;
	
	unsigned int tag;
};


//template<typename Item>
//class LinkedChainIteratorCellWithLayerAccess
//	: public LinkedChainIteratorCell<Item>
//{
//public:
//	LinkedChainIteratorCellWithLayerAccess(LayerGate *layer_gate);
//	
//	LayerGate *GetLayerGate() { return m_layerGate; }
//	
//private:
//	LayerGate *m_layerGate;
//};

}}  // namespace

//include implementation
#include "src/cellLinkedChainIterator.tcc"

#endif /*LINKEDCHAINITERATOR_H_*/
