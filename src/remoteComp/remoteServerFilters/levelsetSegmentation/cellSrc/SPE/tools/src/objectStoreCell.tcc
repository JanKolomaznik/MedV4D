#ifndef OBJECTSTORECELL_H_
#error File objectStoreCell.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////
template<typename T>
ObjectStoreCell<T>
::ObjectStoreCell(uint32 size, T *buf, TAllocMapItem *allocMap)
	: m_allocMap(allocMap), m_buf(buf), m_borrowed(0), _size(size)
{
	// reset the alloc map to zeros
	memset(m_allocMap, 0, sizeof(TAllocMapItem) * ALLOC_MAP_ITEM_COUNT(_size));
}
///////////////////////////////////////////////////////////////////////////////
template<typename T>
T *
ObjectStoreCell<T>::Borrow()
{
	if(m_borrowed < _size)
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
template<typename T> 
void
ObjectStoreCell<T>::Return(T *p)
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

template<typename T> 
void
ObjectStoreCell<T>::ToggleBitInMap(uint16 bitPos)
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
template<typename T>
uint16
ObjectStoreCell<T>::FindFirstFree(void)
{
	TAllocMapItem *it = m_allocMap;
	uint16 cntr = 0;
#define searchMask ((TAllocMapItem)-1) /* full = all ones */
	// search for 0 in alloc map
	while(*it == searchMask	&& cntr < ALLOC_MAP_ITEM_COUNT(_size))
	{
		it++;
		cntr++;
	}
#define BITSINBYTE 8
#define ALLOCITEMSIZEINBITS (sizeof(TAllocMapItem) * BITSINBYTE)
	if(cntr < ALLOC_MAP_ITEM_COUNT(_size))
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

template<typename T, uint16 STORESIZE>
FixedObjectStoreCell<T, STORESIZE>::FixedObjectStoreCell(T *buf)
	: ObjectStoreCell<T>(STORESIZE, buf, m_allocMap)
{

}

///////////////////////////////////////////////////////////////////////////////

#endif
