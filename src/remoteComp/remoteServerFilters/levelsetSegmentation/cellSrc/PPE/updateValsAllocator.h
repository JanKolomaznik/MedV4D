#ifndef UPDATEVALSALLOCATOR_H_
#define UPDATEVALSALLOCATOR_H_

#include <stdlib.h>
#include "../SPE/commonTypes.h"

namespace M4D
{
namespace Cell
{

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
			D_PRINT("UpdateValsAllocator: DEL=" << _array);
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
				D_PRINT("UpdateValsAllocator: DEL=" << _array);
				free(_array);
			}
			
			
//			// aligned to chunks that SPE process
//			_howMany = size + (size % REMOTEARRAY_BUF_SIZE);
			_howMany = (1 + (size / ALLOC_CHUNK_SIZE)) * ALLOC_CHUNK_SIZE;
			
			if( posix_memalign((void **)&_array, 128, _howMany * sizeof(T)) != 0)
				throw "bad";
			
			D_PRINT("UpdateValsAllocator: NEW=" << _array);
		}
		
#ifdef ZERO_BUFFER
		memset(_array, 0, _howMany * sizeof(T));
#endif
	}
private:
	T *_array;
	size_t _howMany;
};

}
}

#endif /*UPDATEVALSALLOCATOR_H_*/
