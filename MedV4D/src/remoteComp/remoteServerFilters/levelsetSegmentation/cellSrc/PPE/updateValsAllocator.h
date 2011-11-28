#ifndef UPDATEVALSALLOCATOR_H_
#define UPDATEVALSALLOCATOR_H_

#include <stdlib.h>
#include "../SPE/commonTypes.h"

namespace M4D
{
namespace Cell
{

#define DEBUG_VALS_ALOCATOR 12

/**
 * Size of one allocation. Prevents repeating allocation of similar size
 */
#define ALLOC_CHUNK_SIZE 256

#define ZERO_BUFFER 1

template<typename T>
class UpdateValsAllocator
{
public:
	UpdateValsAllocator() : _array(NULL), _howMany(0) {}
	~UpdateValsAllocator()
	{
		if(_array)
		{
			DL_PRINT(DEBUG_VALS_ALOCATOR,
					"UpdateValsAllocator: DEL=" << _array);
			free(_array);
		}
	}
	
	T *GetArray() { return _array; }
	
	void AllocArray(size_t size)
	{
		if(_howMany < size)
		{
			if(_array)
			{
				DL_PRINT(DEBUG_VALS_ALOCATOR,
						"UpdateValsAllocator: DEL=" << _array);
				free(_array);
			}
			
			_howMany = (1 + (size / ALLOC_CHUNK_SIZE)) * ALLOC_CHUNK_SIZE;
			
			UConvToVoidPtr uConv; uConv.sp = &_array;
			if( posix_memalign(uConv.vp, 128, _howMany * sizeof(T)) != 0)
				throw std::bad_alloc();
			
			DL_PRINT(DEBUG_VALS_ALOCATOR,
					"UpdateValsAllocator: NEW=" << _array);
		}
		
#ifdef ZERO_BUFFER
		memset(_array, 0, _howMany * sizeof(T));
#endif
	}
private:
	T *_array;
	size_t _howMany;
	
	typedef union {
		T **sp;
		void **vp;
	} UConvToVoidPtr;
};

}
}

#endif /*UPDATEVALSALLOCATOR_H_*/
