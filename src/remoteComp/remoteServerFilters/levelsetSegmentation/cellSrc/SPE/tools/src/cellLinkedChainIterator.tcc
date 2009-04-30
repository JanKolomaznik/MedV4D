#ifndef LINKEDCHAINITERATOR_H_
#error File cellLinkedChainIterator.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {



///////////////////////////////////////////////////////////////////////////////

template<typename Item>
LinkedChainIteratorCell<Item>
::LinkedChainIteratorCell()
		: m_end(0)//, m_begin(0), 
		{
#ifdef FOR_CELL
		/* First, we reserve two MFC tags for use with double buffering */
		  tag = mfc_tag_reserve();
		  if (tag == MFC_TAG_INVALID)
		  {
			  D_PRINT("SPU ERROR, unable to reserve tag\n");
		  }
#endif
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
LinkedChainIteratorCell<Item>::SetBeginEnd(Item *begin, Item *end)
	{
		m_end = end;
		pom = begin;
		
		// load the first item
		m_currBufPosition = 1;	// 1 because its inverted at the begining of next		
		m_buf[1].Next = m_buf;	// to make the first call to HasNext not to return false
		
		if(HasNext())
		{
			m_realAddresses[0] = begin;
//			Load(begin, &m_buf[0], sizeof(Item));
			DMAGate::Get(begin, &m_buf[0], sizeof(Item) );
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
	  mfc_write_tag_mask (1 << tag);
	  mfc_read_tag_status_all ();
#endif
	  m_currBufPosition = ! m_currBufPosition;
	  
	  DL_PRINT(DEBUG_CHAINTOOL, "normal: " << pom->Next 
			  << ", iterator's: " << m_buf[m_currBufPosition].Next );

	  // imediately load the next item
	  if(HasNext())
	  {
		  m_realAddresses[!m_currBufPosition] = GetCurrItem()->Next;
		  //Load(GetCurrItem()->Next, &m_buf[!m_currBufPosition], sizeof(Item));
		  DMAGate::Get(GetCurrItem()->Next, &m_buf[!m_currBufPosition], sizeof(Item) );
		  DL_PRINT(DEBUG_CHAINTOOL, "loading node " << counter);
		  counter++;
	  }
	  
	  pom = pom->Next;
	  
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

}
}

#endif
