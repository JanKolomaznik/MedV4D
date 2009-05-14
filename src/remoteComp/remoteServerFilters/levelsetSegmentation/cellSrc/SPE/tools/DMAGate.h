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

namespace M4D
{
namespace Cell
{

class DMAGate
{
public:
	
#ifdef FOR_CELL
	static unsigned int Put(void *src, Address dest, size_t size)
	{
		unsigned int tag = mfc_tag_reserve();
		if (tag == MFC_TAG_INVALID)
		{
			D_PRINT("SPU ERROR, unable to reserve tag\n");
		}
		mfc_put(src, dest.Get64(), size, tag, 0, 0);
		return tag;
	}
#else
	static void Put(void *src, Address dest, size_t size)
	{
		memcpy((void *)dest.Get64(), src, size);
	}
#endif
	
	
#ifdef FOR_CELL
	static unsigned int Get(Address src, void *dest, size_t size)
	{
		unsigned int tag = mfc_tag_reserve();
		if (tag == MFC_TAG_INVALID)
		{
			D_PRINT("SPU ERROR, unable to reserve tag\n");
		}
		printf ("GET: src=%lld, dest=%p, size=%ld, tag=%d\n", 
				src.Get64(), dest, size, tag);
		mfc_get(dest, src.Get64(), size, tag, 0, 0);
		return tag;
	}
#else
	static void Get(Address src, void *dest, size_t size)
	{
		memcpy(dest, (void *)src.Get64(), size);
	}
#endif
	
};

}
}
#endif /*DMAGATE_H_*/
