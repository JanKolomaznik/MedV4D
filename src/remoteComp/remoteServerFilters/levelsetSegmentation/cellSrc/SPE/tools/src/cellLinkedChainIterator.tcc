#ifndef LINKEDCHAINITERATOR_H_
#error File cellLinkedChainIterator.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
LinkedChainIteratorCell<Item>
::LinkedChainIteratorCell()
		: m_begin(0), m_end(0)
		{
#if( (defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) ) && (NOT_ONLY_TEST) )
		/* First, we reserve two MFC tags for use with double buffering */
		  tag = mfc_tag_reserve();
		  if (tag == MFC_TAG_INVALID)
		  {
			  //D_PRINT ("SPU ERROR, unable to reserve tag\n");
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

//template<typename Item>
//void
//LinkedChainIteratorCell<Item>::SetBeginEnd(Item *begin, Item *end)
//	{
//		m_begin = begin; m_end = end;
//		
//		// load the first item
//		m_currBufPosition = 0;
//		m_nextForLoad = m_currToProcess = begin;
//		
//		if(HasNextForLoad())
//		{
//			Load(m_nextForLoad, &m_buf[m_currBufPosition], sizeof(Item));
//			m_nextForLoad = begin->Next;
//		}
//	}
//
/////////////////////////////////////////////////////////////////////////////////
//
//template<typename Item>
//Item *
//LinkedChainIteratorCell<Item>::Next(void) 
//	{ 
//#if( (defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) ) && (NOT_ONLY_TEST) )
//	  // wait for current DMA to complete
//	  mfc_write_tag_mask (1 << tag);
//	  mfc_read_tag_status_all ();
//#endif
//		  
//	  // imediately load the next item
//	  m_currBufPosition = ! m_currBufPosition;
//	  
//	  if(HasNextForLoad())
//	  {
//		  Load(m_nextForLoad, &m_buf[m_currBufPosition], sizeof(Item));
//		  m_nextForLoad = m_nextForLoad->Next;
//	  }
//	  
//	  m_currToProcess = m_currToProcess->Next;
//	  return &m_buf[! m_currBufPosition]; 
//	  }

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
void
LinkedChainIteratorCell<Item>::Load(Item *src, Item *dest, size_t size)
	{
#if( (defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) ) && (NOT_ONLY_TEST) )
			mfc_get(src, dest, size, tag, 0, 0);
#else
			memcpy(dest, src, size);
#endif
	}

///////////////////////////////////////////////////////////////////////////////

template<typename Item>
LinkedChainIteratorCellWithLayerAccess<Item>
::LinkedChainIteratorCellWithLayerAccess(LayerGate *layer_gate)
	: m_layerGate(m_layerGate)
{

}

///////////////////////////////////////////////////////////////////////////////

}
}

#endif
