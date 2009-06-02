#ifndef LINKEDCHAINITERATOR_H_
#error File cellLinkedChainIterator.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
LinkedChainIteratorCell<Item>
::LinkedChainIteratorCell()
: m_end(0)
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
	m_buf[1].Next = Address((uint64)m_buf); // to make the first call to HasNext not to return false

	if(HasNext())
	{
		m_realAddresses[0] = begin;
#ifdef FOR_CELL
		tag = DMAGate::GetTag();
#endif
		DMAGate::Get(begin, &m_buf[0], sizeof(Item), tag);
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_GET:LinkedChainIteratorCell:%d\n", tag);
#endif
		DL_PRINT(DEBUG_CHAINTOOL, "loading the first node in chain \n");
		counter = 1;
	}
	else
	{
		m_buf[!m_currBufPosition].Next = 0;
	}

}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
Item *
LinkedChainIteratorCell<Item>::Next(void)
{
#ifdef FOR_CELL
	// wait for current DMA to complete
	mfc_write_tag_mask(1 << tag);
	mfc_read_tag_status_all();
#endif
	m_currBufPosition = ! m_currBufPosition;

#ifdef FOR_CELL
	DL_PRINT(DEBUG_CHAINTOOL, "normal: %lld, iterator's: %lld"
			,( (Item *)pom.Get64() )->Next.Get64(),
			m_buf[m_currBufPosition].Next.Get64() );
#else
	DL_PRINT(DEBUG_CHAINTOOL, "normal: " << ( (Item *)pom.Get64() )->Next.Get64()
			<< ", iterator's: " << m_buf[m_currBufPosition].Next.Get64() );
#endif

	// imediately load the next item
	if(HasNext())
	{
		m_realAddresses[!m_currBufPosition] = GetCurrItem()->Next;
		DMAGate::Get(GetCurrItem()->Next, &m_buf[!m_currBufPosition], sizeof(Item), tag);		
#ifdef FOR_CELL
		DL_PRINT(DEBUG_CHAINTOOL, "loading node %d", counter);
#else
		DL_PRINT(DEBUG_CHAINTOOL, "loading node " << counter);
#endif
		counter++;
	}
#ifdef FOR_CELL
	else
	{
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_RET:LinkedChainIteratorCell:%d\n", tag);
#endif
		//return tag
		DMAGate::ReturnTag(tag);
	}
#endif

	pom = ( (Item *)pom.Get64() )->Next;

	return GetCurrItem();
}

///////////////////////////////////////////////////////////////////////////////

#endif
