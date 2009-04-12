#ifndef OBJECTSTORECELL_H_
#define OBJECTSTORECELL_H_

namespace M4D
{
namespace Cell
{

template<typename T, uint16 STORESIZE>
class ObjectStoreCell
{
public:
	ObjectStoreCell();
	
	 T *Borrow();
	 void Return(T *p);
	 
	 bool IsFull() { return m_borrowed == STORESIZE; }
	
private:
	void ToggleBitInMap(uint16 bitPos);
	uint16 FindFirstFree(void);
	
	typedef uint8 TAllocMapItem;
#define ALLOC_MAP_ITEM_SIZE_IN_BITS (sizeof(TAllocMapItem) * 8)
#define ALLOC_MAP_ITEM_COUNT ((STORESIZE/ALLOC_MAP_ITEM_SIZE_IN_BITS) + 1)
	TAllocMapItem m_allocMap[ALLOC_MAP_ITEM_COUNT];
	T m_buf[STORESIZE];
	
	uint32 m_borrowed;
};

}
}

//include implementation
#include "src/objectStoreCell.tcc"

#endif /*OBJECTSTORECELL_H_*/
