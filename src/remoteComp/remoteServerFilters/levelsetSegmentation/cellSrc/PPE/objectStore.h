#ifndef OBJECTSTORE_H_
#define OBJECTSTORE_H_

#include "../SPE/tools/objectStoreCell.h"
#include <vector>
#include <stdlib.h>

namespace M4D
{
namespace Cell
{

template<typename T, uint16 CHUNKSIZE>
class PPEObjectStore
{
public:
	typedef FixedObjectStoreCell<T, CHUNKSIZE> TChunk;
	typedef std::vector<T*> VecChunksBegin;
	typedef std::vector<TChunk*> VecChunks;
	
	~PPEObjectStore()
	{		
		for(typename VecChunksBegin::iterator it=_chunksBegin.begin();
			it != _chunksBegin.end(); it++)
		{
			free(*it);
		}
		for(typename VecChunks::iterator it=_chunks.begin();
			it != _chunks.end(); it++)
		{
			free(*it);
		}
	}
	
	T *Borrow()
	{
		if(_freeChunks.empty())
			AllocNewChunk();
		TChunk *chunk = _freeChunks.back();
		T *tmp = chunk->Borrow();
		if(chunk->IsFull())
			_freeChunks.pop_back();
		return tmp;
	}
	
	void Return(T *p)
	{
		TChunk *chunk = FindItemsChunk(p);
		if(chunk->IsFull())
			_freeChunks.push_back(chunk);
		chunk->Return(p);
	}
protected:
	
	void AllocNewChunk()
	{
		T *newArray = NULL;
		if( posix_memalign((void**)(&newArray), 128, CHUNKSIZE * sizeof(T)) != 0)
			throw std::bad_alloc();
		TChunk *newChunk = new TChunk(newArray);
		_chunks.push_back(newChunk);
		_freeChunks.push_back(newChunk);
		_chunksBegin.push_back(newArray);
	}
	
	TChunk *FindItemsChunk(T *item)
	{
		uint32 pos = 0;
		typename VecChunksBegin::iterator it = _chunksBegin.begin();
		while(it != _chunksBegin.end())
		{
			if((item >= *it) && (item < (*it + CHUNKSIZE)) )
			{
				return _chunks[pos];
			}
			it++; pos++;
		}
		return NULL;
	}
	
	VecChunks _freeChunks;
	VecChunks _chunks;
	VecChunksBegin _chunksBegin;
};

}
}
#endif /*OBJECTSTORE_H_*/