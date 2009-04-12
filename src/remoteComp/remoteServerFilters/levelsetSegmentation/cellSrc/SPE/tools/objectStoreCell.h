#ifndef OBJECTSTORECELL_H_
#define OBJECTSTORECELL_H_

namespace M4D
{
namespace Cell
{

template<typename T, uint16 SIZE>
class ObjectStoreCell
{
	 T *Borrow();
	 void Return(T *p);
	
private:
	void ToggleBitInMap(uint16 bitPos);
	uint16 FindFirstFree(void);
	
	typedef uint64 TAllocMapItem;
#define ALLOC_MAP_ITEM_COUNT (SIZE/sizeof(TAllocMapItem))
	TAllocMapItem m_allocMap[ALLOC_MAP_ITEM_COUNT];
	T m_buf[SIZE];
	
	bool m_full;
};

}
}

//include implementation
#include "src/objectStoreCell.tcc"
#endif /*OBJECTSTORECELL_H_*/
