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

#define DEBUG_MFC 1

namespace M4D
{
namespace Cell
{

class DMAGate
{
public:

	///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
	static unsigned int Put(void *src, Address dest, size_t size)
	{
		unsigned int tag = GetTag();
		mfc_put(src, dest.Get64(), size, tag, 0, 0);
		return tag;
	}
#else
	static void Put(void *src, Address dest, size_t size)
	{
		memcpy((void *)dest.Get64(), src, size);
	}
#endif
	///////////////////////////////////////////////////////////////////////////////

#ifdef FOR_CELL
	static unsigned int GetList(uint64 src, void *dest,
			mfc_list_element_t *list, size_t listSize)
	{
		unsigned int tag = GetTag();
		printf(
				"GETL: src=%lX, dest=%p, listSize=%lu, listSizePutInto=%ld, tag=%d\n",
				(long unsigned int) src, dest, listSize, listSize
						* sizeof(mfc_list_element_t), tag);
		uint32 totalSize=0;
		for (uint32 i=0; i<listSize; i++)
		{
			printf("\t LEAL=%X, size=%lld", (uint32) list[i].eal,
					list[i].size);
			// check aligns
			if ( ( (list[i].size == 2) && ((list[i].eal & 0x1) != 0) )
				|| ( (list[i].size == 4) && ((list[i].eal & 0x3) != 0) ) 
				|| ( (list[i].size == 8) && ((list[i].eal & 0x7) != 0) )
				|| ( (list[i].size >= 16)&& ((list[i].eal & 0xF) != 0) ))
			{
				printf(" = Wrong align!!\n");
			}
			else
				printf("\n");
			totalSize += list[i].size;
		}
		printf("total size=%u\n", totalSize);
#ifdef DEBUG_MFC
		if (((uint32)list & 0x7) != 0)
		{
			printf("Wrong list address!!");
		}
#endif
		mfc_getl(dest, src, list, listSize * sizeof(mfc_list_element_t), tag,
				0, 0);
		return tag;
	}

	///////////////////////////////////////////////////////////////////////////////

	static unsigned int PutList(uint64 dest, void *locaBuf,
			mfc_list_element_t *list, size_t listSize)
	{
		unsigned int tag = GetTag();
		printf(
				"PUTL: localBuf=%p, EA_dest=%lx, listSize=%lu, listSizePutInto=%lu, tag=%d\n",
				locaBuf, (long unsigned int) dest, listSize, listSize
						* sizeof(mfc_list_element_t), tag);
		uint32 totalSize=0;
		for (uint32 i=0; i<listSize; i++)
		{
			printf("\t LEAL=%X, size=%lld\n", (uint32) list[i].eal,
					list[i].size);
			totalSize += list[i].size;
		}
		printf("total size=%u\n", totalSize);
#ifdef DEBUG_MFC
		if (((uint32)list & 0x7) != 0)
		{
			printf("Wrong list address!!");
		}
#endif
		mfc_putl(locaBuf, dest, list, listSize * sizeof(mfc_list_element_t), tag,
				0, 0);
		return tag;
	}

	///////////////////////////////////////////////////////////////////////////////

	static unsigned int Get(Address src, void *dest, size_t size)
	{
		unsigned int tag = GetTag();
#ifdef DEBUG_MFC
		printf("GET: src=%p, dest=%p, size=%ld, tag=%d\n", (void*)src.Get64(),
				dest, size, tag);

		// check trasfer size
		if ( ! ( (size == 1) || (size == 2) || (size == 4) || (size == 8)
				|| ((size % 16) == 0) ))
		{
			printf("Wrong trasfer size!!");
		}
#define MAX_TRANSFER_SIZE (1024 * 16)	// 16 kB
		if (size > MAX_TRANSFER_SIZE)
		{
			printf("Trasfer size too long!!");
		}

		// check aligns
		if ( ( (size == 2) && (((uint32)dest & 0x1) != 0) ) || ( (size == 4)
				&& (((uint32)dest & 0x3) != 0) ) || ( (size == 8)
				&& (((uint32)dest & 0x7) != 0) ) || ( (size >= 16)
				&& (((uint32)dest & 0xF) != 0) ))
		{
			printf("Wrong align!!");
		}
		if ((src.Get64() & 0x7) != ((uint32)dest & 0x7))
		{
			printf("Wrong align address tails not equal!!");
		}
#endif
		mfc_get(dest, src.Get64(), size, tag, 0, 0);
#if DEBUG_MFC
		// When debugging we can force the transfer to complete immediately to keep things simple
		mfc_write_tag_mask(1 << tag);
		mfc_read_tag_status_all();
#endif
		return tag;
	}
#else
	static void Get(Address src, void *dest, size_t size)
	{
		memcpy(dest, (void *)src.Get64(), size);
	}
#endif

#ifdef FOR_CELL

	static void ReturnTag(uint32 tag)
	{
		mfc_tag_release(tag);
	}
private:
	static unsigned int GetTag()
	{
		unsigned int tag = mfc_tag_reserve();
		if (tag == MFC_TAG_INVALID)
		{
			D_PRINT("SPU ERROR, unable to reserve tag\n");
		}
		return tag;
	}

#endif
};

}
}
#endif /*DMAGATE_H_*/
