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
	loadedIter = processedIter = begin;

	// load the first item
	//m_currBufPosition = 1; // 1 because its inverted at the begining of next
	m_currBufPosition = _loadingPos = _loadedPos = 0;
	m_buf[_loadedPos].Next = begin;
//	m_buf[1].Next = Address((uint64)m_buf); // to make the first call to HasNext not to return false
	//m_buf[!m_currBufPosition].Next = 0;
	
	if(loadedIter != end)
	{
		m_realAddresses[_loadingPos] = begin;
		//			Load(begin, &m_buf[0], sizeof(Item));
#ifdef FOR_CELL
		DMAGate::Get(begin, &m_buf[_loadingPos], sizeof(Item), _tag);
#else
		DMAGate::Get(begin, &m_buf[_loadingPos], sizeof(Item) );
#endif
		DL_PRINT(DEBUG_CHAINTOOL, "loading the first node in chain \n");
		counter = 1;
		//loadedIter = ( (Item *)loadedIter.Get64() )->Next;
	}
//	else
//	{
//		m_buf[!m_currBufPosition].Next = 0;
//	}
	//return begin != end;	// we have put something to load	
}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
Item *
LinkedChainIteratorCell<Item>::GetLoaded(void)
{
#ifdef FOR_CELL
	// wait for current DMA to complete
	mfc_write_tag_mask (1 << _tag);
	mfc_read_tag_status_all ();
#endif
	//m_currBufPosition = ! m_currBufPosition;

#ifdef FOR_CELL
	DL_PRINT(DEBUG_CHAINTOOL, "normal: %lld, iterator's: %lld"
			,( (Item *)loadedIter.Get64() )->Next.Get64(),
			m_buf[_loadingPos].Next.Get64() );
#else
	DL_PRINT(DEBUG_CHAINTOOL, "normal: " << ( (Item *)loadedIter.Get64() )->Next.Get64()
			<< ", iterator's: " << m_buf[_loadingPos].Next.Get64() );
#endif
	
	_loadedPos = _loadingPos;

	// imediately load the next item
	if(loadedIter != m_end)
	{
		_loadingPos = (_loadingPos + 1) % BUF_SIZE;
		loadedIter = m_buf[_loadedPos].Next;
		
		m_realAddresses[_loadingPos] = m_buf[_loadedPos].Next;
		//Load(GetCurrItem()->Next, &m_buf[!m_currBufPosition], sizeof(Item));
		
#ifdef FOR_CELL
		DL_PRINT(DEBUG_CHAINTOOL, "loading node %d", counter);
		DMAGate::Get(m_buf[_loadedPos].Next, &m_buf[_loadingPos], sizeof(Item), _tag);
#else
		DL_PRINT(DEBUG_CHAINTOOL, "loading node " << counter);
		DMAGate::Get(m_buf[_loadedPos].Next, &m_buf[_loadingPos], sizeof(Item) );
#endif
		counter++;
	}

//	processedIter = ( (Item *)processedIter.Get64() )->Next;

	return &m_buf[_loadedPos];
}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
void
LinkedChainIteratorCell<Item>::Next(void)
{
	m_currBufPosition = (m_currBufPosition + 1) % BUF_SIZE;;
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL

template<typename Item>
void
LinkedChainIteratorCell<Item>::ReserveTag()
{
#ifdef TAG_RETURN_DEBUG
	_tag = DMAGate::GetTag();
	D_PRINT("TAG_GET:LinkedChainIteratorCell:%d\n", _tag);
#endif
}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
void
LinkedChainIteratorCell<Item>::ReturnTag()
{
#ifdef TAG_RETURN_DEBUG
	DMAGate::ReturnTag(_tag);
	D_PRINT("TAG_RET:LinkedChainIteratorCell:%d\n", _tag);
#endif
}

#endif // FOR_CELL
///////////////////////////////////////////////////////////////////////////////

#endif // whole file
