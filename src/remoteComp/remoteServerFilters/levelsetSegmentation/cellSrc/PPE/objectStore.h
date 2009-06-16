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
	
	PPEObjectStore()
		//: borrowed(0), maxBorrowed(0) 
		{}
	
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
			delete(*it);
		}
	}
	
	T *Borrow()
	{
		//borrowed++;
		//if(borrowed > maxBorrowed)
		//	maxBorrowed = borrowed;
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
		//borrowed--;
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
	
	void Print(std::ostream &str)
	{
		uint cnt=0;
		str << "Chunks: " << std::endl;
		for(typename VecChunks::iterator it=_chunks.begin();
			it != _chunks.end(); it++)
		{
			str << *it << ",_chunks[" << cnt << "]=" << _chunks[cnt] << std::endl;
			cnt++;
		}
		str << "ChunkBegins: " << std::endl;
		cnt = 0;
		for(typename VecChunksBegin::iterator it=_chunksBegin.begin();
			it != _chunksBegin.end(); it++)
		{
			str << *it << ",_chunksBegin[" << cnt << "]=" << _chunksBegin[cnt] << std::endl;
			cnt++;
		}
	}
	
protected:
	
	typedef union {
		T **sp;
		void **vp;
	} UConvToVoidPtr;
	
	void AllocNewChunk()
	{
		T *newArray = NULL;
		UConvToVoidPtr uConv; uConv.sp = &newArray;
		if( posix_memalign(uConv.vp, 128, CHUNKSIZE * sizeof(T)) != 0)
			throw std::bad_alloc();
		TChunk *newChunk = new TChunk(newArray);
		_chunks.push_back(newChunk);
		_freeChunks.push_back(newChunk);
		_chunksBegin.push_back(newArray);
	}
	
//	uint32 borrowed;
//	uint32 maxBorrowed;
	
	VecChunks _freeChunks;
	VecChunks _chunks;
	VecChunksBegin _chunksBegin;
};

}
}
#endif /*OBJECTSTORE_H_*/
