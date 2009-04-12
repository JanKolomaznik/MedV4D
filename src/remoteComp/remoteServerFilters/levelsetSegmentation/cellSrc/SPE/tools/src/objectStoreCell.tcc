#ifndef OBJECTSTORECELL_H_
#error File objectStoreCell.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE>
T *
ObjectStoreCell<T, STORESIZE>::Borrow()
{
	uint16 pos = FindFirstFree();
	ToggleBitInMap(pos);
	return m_buf[pos];
}	 

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE> 
void
ObjectStoreCell<T, STORESIZE>::Return(T *p)
{
	// only update alloc map, position is counted from address
	ToggleBitInMap( (p - &m_buf) / sizeof(T));
}

///////////////////////////////////////////////////////////////////////////////
#define TOGGLE_BIT(cell, mask)  ((cell) = (cell) ^ (mask)) 

template<typename T, uint16 STORESIZE> 
void
ObjectStoreCell<T, STORESIZE>::ToggleBitInMap(uint16 bitPos)
{
	uint16 cellNum = bitPos / sizeof(TAllocMapItem);
	TAllocMapItem *cell = &m_allocMap[cellNum];
	
	// create mask used to set the bit
	TAllocMapItem mask = 1;
//#define IN_MASK_POS (bitPos % sizeof(TAllocMapItem))
	uint32 IN_MASK_POS = bitPos % sizeof(TAllocMapItem);
	cellNum = 0;	// reusing
	while(cellNum < IN_MASK_POS)
		mask = mask << 1;
	
	TOGGLE_BIT(*cell,mask);
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE>
uint16
ObjectStoreCell<T, STORESIZE>::FindFirstFree(void)
{
	TAllocMapItem *it = &m_allocMap;
	uint16 cntr = 0;
	while(*it == 1 && cntr < ALLOC_MAP_ITEM_COUNT)	// search for 0 in alloc map
		;
	
	if(cntr < ALLOC_MAP_ITEM_COUNT)
	{
		uint16 retval = cntr * sizeof(TAllocMapItem);
		// add bit count in found cell
		TAllocMapItem tmp = *it;
		while(tmp & 1)
		{
			cntr++;
			tmp >> 1;	// observe next bit
		}
		return cntr;
	}
	else
	{
		m_full = true;
		return 65535;
	}
}

///////////////////////////////////////////////////////////////////////////////

}
}
#endif
