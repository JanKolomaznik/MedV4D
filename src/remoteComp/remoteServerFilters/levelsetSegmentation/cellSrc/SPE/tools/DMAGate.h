#ifndef DMAGATE_H_
#define DMAGATE_H_

#include "address.h"

#ifdef FOR_CELL
#include <spu_mfcio.h>
#include "SPEdebug.h"
#else
#include "common/Debug.h"
#include <string.h>
#endif

//#define DEBUG_MFC 1
//#define TAG_RETURN_DEBUG 1

namespace M4D
{
namespace Cell
{

class DMAGate
{
#ifdef FOR_CELL
public:

	static inline void Put(void *src, Address dest, size_t size, uint32 tag)
	{
#ifdef DEBUG_MFC
		D_PRINT("PUT: LA=%p, EA=0x%llx, size=%ld, tag=%u\n", src,
						dest.Get64(), size, tag);		
		CheckTransferSize(size);
#endif
		D_COMMAND(CheckTag(tag));
		mfc_put(src, dest.Get64(), size, tag, 0, 0);
	}
	
	///////////////////////////////////////////////////////////////////////////////

	static void GetList(uint64 src, void *dest,
			mfc_list_element_t *list, size_t listSize, uint32 tag)
	{
#ifdef DEBUG_MFC
		D_PRINT(
				"GETL: src=%lX, dest=%p, listSize=%lu, listSizePutInto=%ld, tag=%u\n",
				(long unsigned int) src, dest, listSize, listSize
						* sizeof(mfc_list_element_t), tag);
		PrintListContent(list, listSize, true);
		CheckLSListAddress(list);
#endif
		D_COMMAND(CheckTag(tag));
		mfc_getl(dest, src, list, listSize * sizeof(mfc_list_element_t), tag,
				0, 0);
	}

	///////////////////////////////////////////////////////////////////////////////

	static void PutList(uint64 dest, void *locaBuf,
			mfc_list_element_t *list, size_t listSize, uint32 tag)
	{
#ifdef DEBUG_MFC
		D_PRINT("PUTL: localBuf=%p, EA_dest=%lx, listSize=%lu, listSizePutInto=%lu, tag=%u\n",
						locaBuf, (long unsigned int) dest, listSize, listSize
								* sizeof(mfc_list_element_t), tag);
		PrintListContent(list, listSize, false);
		CheckLSListAddress(list);
#endif
		D_COMMAND(CheckTag(tag));
		mfc_putl(locaBuf, dest, list, listSize * sizeof(mfc_list_element_t), tag,
				0, 0);
	}

	///////////////////////////////////////////////////////////////////////////////

	static void Get(Address src, void *dest, size_t size, uint32 tag)
	{
#ifdef DEBUG_MFC
		D_PRINT("GET: src=%p, dest=%p, size=%ld, tag=%u\n", (void*)src.Get64(),
				dest, size, tag);
		CheckTransferSize(size);
		CheckAlign(src, dest, size);		
#endif
		D_COMMAND(CheckTag(tag));
		mfc_get(dest, src.Get64(), size, tag, 0, 0);
//#if DEBUG_MFC
//		// When debugging we can force the transfer to complete immediately to keep things simple
//		mfc_write_tag_mask(1 << tag);
//		mfc_read_tag_status_all();
//#endif
	}

	static void ReturnTag(uint32 tag)
	{
		mfc_tag_release(tag);
	}
	
	static unsigned int GetTag()
	{
		unsigned int tag = mfc_tag_reserve();
		if (tag == MFC_TAG_INVALID)
		{
			D_PRINT("SPU ERROR, unable to reserve tag\n");
		}
		return tag;
	}

private:
	
	static bool CheckTag(int32 tag)
	{
		if(tag >= 32)
		{
			D_PRINT("Using wrong tag = %d\n", tag);
			return false;
		}
		return true;
	}
	
	static bool CheckTransferSize(size_t size)
	{
		bool ok = true;
		// check trasfer size
		if ( ! ( (size == 1) || (size == 2) || (size == 4) || (size == 8)
				|| ((size % 16) == 0) ))
		{
			D_PRINT("Wrong trasfer size!!");
			ok = false;
		}
#define MAX_TRANSFER_SIZE (1024 * 16)	// 16 kB
		if (size > MAX_TRANSFER_SIZE)
		{
			D_PRINT("Trasfer size too long!!");
			ok = false;
		}
		return ok;
	}
	
	static bool CheckAlign(Address src, void *dest, size_t size)
	{
		bool ok = true;
		// check aligns
		if ( ( (size == 2) && (((uint32)dest & 0x1) != 0) ) || ( (size == 4)
				&& (((uint32)dest & 0x3) != 0) ) || ( (size == 8)
				&& (((uint32)dest & 0x7) != 0) ) || ( (size >= 16)
				&& (((uint32)dest & 0xF) != 0) ))
		{
			D_PRINT("Wrong align!!");
			ok = false;
		}
		if ((src.Get64() & 0x7) != ((uint32)dest & 0x7))
		{
			D_PRINT("Wrong align address tails not equal!!");
			ok = false;
		}
		return ok;
	}
	
	/**
	 * Check if address of list on LocalStore is 32byte aligned
	 */
	static bool CheckLSListAddress(void *list)
	{
		if (((uint32)list & 0x7) != 0)
		{
			D_PRINT("Wrong list address!!");
			return false;
		}
		return true;
	}
	
	static bool PrintListContent(mfc_list_element_t *list, size_t listSize, bool checkAlign)
	{
		uint32 totalSize=0;
		bool ok = true;
		for (uint32 i=0; i<listSize; i++)
		{
			D_PRINT("\t LEAL=%X, size=%lld", (uint32) list[i].eal,
					list[i].size);
			
			if(checkAlign)
			{
				// check aligns
				if ( ( (list[i].size == 2) && ((list[i].eal & 0x1) != 0) )
					|| ( (list[i].size == 4) && ((list[i].eal & 0x3) != 0) ) 
					|| ( (list[i].size == 8) && ((list[i].eal & 0x7) != 0) )
					|| ( (list[i].size >= 16)&& ((list[i].eal & 0xF) != 0) ))
				{
					D_PRINT(" = Wrong align!!\n");
					ok = false;
				}
				else
					D_PRINT("\n");
			}
			totalSize += list[i].size;
		}
		D_PRINT("total size=%u\n", totalSize);
		return ok;
	}
	
	///////////////////////////////////////////////////////////////////////////
	
#else	/* FOR_PC */
	
public:
	static void Put(void *src, Address dest, size_t size)
	{
		memcpy((void *)dest.Get64(), src, size);
		//tag++; // to prevent warns
	}
	
	static void Get(Address src, void *dest, size_t size)
	{
		memcpy(dest, (void *)src.Get64(), size);
		//tag++;	// to prevent warns
	}	
#endif
};

}
}
#endif /*DMAGATE_H_*/
