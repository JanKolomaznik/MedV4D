#ifndef OBJECTSTORECELL_H_
#error File objectStoreCell.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE>
ObjectStoreCell<T, STORESIZE>::ObjectStoreCell()
	: m_borrowed(0)
{
	
}
///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE>
T *
ObjectStoreCell<T, STORESIZE>::Borrow()
{
	if(m_borrowed < STORESIZE)
	{
		uint16 pos = FindFirstFree();
		ToggleBitInMap(pos);
		m_borrowed++;
		//D_PRINT("borrowing: " << &m_buf[pos] << " on " << pos << ", borrowed=" << m_borrowed);
		return &m_buf[pos];
	}
	return NULL;
}	 

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE> 
void
ObjectStoreCell<T, STORESIZE>::Return(T *p)
{
//	if(p < m_buf || p >= &m_buf[STORESIZE] )	// trying put foreign node
//	{
//		D_PRINT("PUTTING FOREIGN!");
//		return;
//	}
	
	// only update alloc map, position is counted from address
#define POS (p - m_buf)
	
	ToggleBitInMap(POS);
	m_borrowed--;
	//D_PRINT("returning: " << p << " on " << pos << ", borrowed=" << m_borrowed);
}

///////////////////////////////////////////////////////////////////////////////
#define TOGGLE_BIT(cell, mask)  ((cell) = (cell) ^ (mask)) 

template<typename T, uint16 STORESIZE> 
void
ObjectStoreCell<T, STORESIZE>::ToggleBitInMap(uint16 bitPos)
{
	uint16 cellNum = bitPos / ALLOC_MAP_ITEM_SIZE_IN_BITS;
	TAllocMapItem *cell = &m_allocMap[cellNum];
	
	// create mask used to set the bit
	TAllocMapItem mask = 1;
#define IN_MASK_POS (bitPos % ALLOC_MAP_ITEM_SIZE_IN_BITS)
	cellNum = 0;	// reusing
	while(cellNum < IN_MASK_POS)
	{
		mask <<= 1;
		cellNum++;
	}
	
	TOGGLE_BIT(*cell,mask);
}

///////////////////////////////////////////////////////////////////////////////
template<typename T, uint16 STORESIZE>
uint16
ObjectStoreCell<T, STORESIZE>::FindFirstFree(void)
{
	TAllocMapItem *it = m_allocMap;
	uint16 cntr = 0;
#define searchMask ((TAllocMapItem)-1) /* full = all ones */
	// search for 0 in alloc map
	while(*it == searchMask	&& cntr < ALLOC_MAP_ITEM_COUNT)
	{
		it++;
		cntr++;
	}
#define BITSINBYTE 8
#define ALLOCITEMSIZEINBITS (sizeof(TAllocMapItem) * BITSINBYTE)
	if(cntr < ALLOC_MAP_ITEM_COUNT)
	{
		cntr *= ALLOCITEMSIZEINBITS;
		// add bit count in found cell
		TAllocMapItem tmp = *it;
		while(tmp & 1)
		{
			cntr++;
			tmp >>= 1;	// observe next bit
		}
		return cntr;
	}
	return (uint16)-1;
}

///////////////////////////////////////////////////////////////////////////////

}
}
#endif
