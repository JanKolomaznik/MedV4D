#ifndef LINKEDCHAINITERATOR_H_
#define LINKEDCHAINITERATOR_H_


namespace M4D {
namespace Cell {

template<typename Item>
class LinkedChainIteratorCell
{
public:
	LinkedChainIteratorCell()
		: m_begin(0), m_end(0)
		{
#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
		/* First, we reserve two MFC tags for use with double buffering */
		  tag = mfc_tag_reserve();
		  if (tag == MFC_TAG_INVALID)
		  {
			  D_PRINT ("SPU ERROR, unable to reserve tag\n");
		  }
#endif
		}
	
	void SetBeginEnd(Item *begin, Item *end)
	{
		m_begin = begin; m_end = end;
		
		// load the first item
		m_currBufPosition = 0;
		m_nextForLoad = m_currToProcess = begin;
		
		if(HasNextForLoad())
		{
			Load(m_nextForLoad, &m_buf[m_currBufPosition], sizeof(Item));
			m_nextForLoad = begin->Next;
		}
	}
	
	bool HasNext(void) { return (m_currToProcess != m_end); }
	
	Item *Next(void) 
	{ 
	// wait for current DMA to complete
//	  mfc_write_tag_mask (1 << tag);
//	  mfc_read_tag_status_all ();
		  
	  // imediately load the next item
	  m_currBufPosition = ! m_currBufPosition;
	  
	  if(HasNextForLoad())
	  {
		  Load(m_nextForLoad, &m_buf[m_currBufPosition], sizeof(Item));
		  m_nextForLoad = m_nextForLoad->Next;
	  }
	  
	  m_currToProcess = m_currToProcess->Next;
	  return &m_buf[! m_currBufPosition]; 
	  }
private:
	
	bool HasNextForLoad(void) { return (m_nextForLoad != m_end); }
	
	void Load(Item *src, Item *dest, size_t size)
	{
#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
			mfc_get(src, dest, sizeof(Item), tag, 0, 0);
#else
			memcpy(dest, src, sizeof(Item) );
#endif
	}
	
	Item m_buf[2];
	uint8 m_currBufPosition;
	
	Item *m_begin, *m_end;
	Item *m_nextForLoad;
	Item *m_currToProcess;
	
	unsigned int tag;	  
};

}}  // namespace

#endif /*LINKEDCHAINITERATOR_H_*/
