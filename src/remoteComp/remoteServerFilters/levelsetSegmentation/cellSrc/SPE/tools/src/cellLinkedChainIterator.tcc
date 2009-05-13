#ifndef LINKEDCHAINITERATOR_H_
#error File cellLinkedChainIterator.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
LinkedChainIteratorCell<Item>
::LinkedChainIteratorCell()
: m_end(0)//, m_begin(0), 

{
}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
LinkedChainIteratorCell<Item>
::~LinkedChainIteratorCell()
{

}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
void
LinkedChainIteratorCell<Item>::SetBeginEnd(Address begin, Address end)
{
	m_end = end;
	pom = begin;

	// load the first item
	m_currBufPosition = 1; // 1 because its inverted at the begining of next		
	m_buf[1].Next = Address(m_buf); // to make the first call to HasNext not to return false

	if(HasNext())
	{
		m_realAddresses[0] = begin;
		//			Load(begin, &m_buf[0], sizeof(Item));
#ifdef FOR_CELL
		tag = DMAGate::Get(begin, &m_buf[0], sizeof(Item) );
#else
		DMAGate::Get(begin, &m_buf[0], sizeof(Item) );
#endif
		DL_PRINT(DEBUG_CHAINTOOL, "loading the first node in chain \n");
		counter = 1;
	}
	else
	{
		m_buf[!m_currBufPosition].Next = NULL;
	}

}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
Item *
LinkedChainIteratorCell<Item>::Next(void)
{
#ifdef FOR_CELL
	// wait for current DMA to complete
	mfc_write_tag_mask (1 << tag);
	mfc_read_tag_status_all ();
#endif
	m_currBufPosition = ! m_currBufPosition;

	DL_PRINT(DEBUG_CHAINTOOL, "normal: " << ( (Item *)pom.Get64() )->Next.Get64()
			<< ", iterator's: " << m_buf[m_currBufPosition].Next.Get64() );

	// imediately load the next item
	if(HasNext())
	{
		m_realAddresses[!m_currBufPosition] = GetCurrItem()->Next;
		//Load(GetCurrItem()->Next, &m_buf[!m_currBufPosition], sizeof(Item));
		DMAGate::Get(GetCurrItem()->Next, &m_buf[!m_currBufPosition], sizeof(Item) );
		DL_PRINT(DEBUG_CHAINTOOL, "loading node " << counter);
		counter++;
	}

	pom = ( (Item *)pom.Get64() )->Next;

	return GetCurrItem();
}

///////////////////////////////////////////////////////////////////////////////

//template<typename Item>
//void
//LinkedChainIteratorCell<Item>::Load(Item *src, Item *dest, size_t size)
//	{
//#ifdef FOR_CELL
//			mfc_get(src, dest, size, tag, 0, 0);
//#else
//			memcpy(dest, src, size);
//#endif
//	}

///////////////////////////////////////////////////////////////////////////////

#endif
