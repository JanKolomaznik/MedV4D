#ifndef OBJECTSTORECELL_H_
#define OBJECTSTORECELL_H_

#include <string.h>

namespace M4D
{
namespace Cell
{

typedef uint64 TAllocMapItem;

template<typename T>
class ObjectStoreCell
{
public:	
	 T *Borrow();
	 void Return(T *p);
	 
	 bool IsFull() { return m_borrowed == _size; }		
	ObjectStoreCell(uint32 size, T *buf, TAllocMapItem *allocMap);
private:
	void ToggleBitInMap(uint16 bitPos);
	uint16 FindFirstFree(void);
	
	
#define ALLOC_MAP_ITEM_SIZE_IN_BITS (sizeof(TAllocMapItem) * 8)
#define ALLOC_MAP_ITEM_COUNT(SIZE) ((SIZE/ALLOC_MAP_ITEM_SIZE_IN_BITS) + 1)
	
	TAllocMapItem *m_allocMap;
	T *m_buf;
	
	uint32 m_borrowed;
	uint32 _size;
};

///////////////////////////////////////////////////////////////////////////////

template<typename T, uint16 STORESIZE>
class FixedObjectStoreCell : public ObjectStoreCell<T>
{
public:
	FixedObjectStoreCell(T *buf);
private:
	TAllocMapItem m_allocMap[ALLOC_MAP_ITEM_COUNT(STORESIZE)];
};

///////////////////////////////////////////////////////////////////////////////

//include implementation
#include "src/objectStoreCell.tcc"

}
}

#endif /*OBJECTSTORECELL_H_*/
