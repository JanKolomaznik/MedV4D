#ifndef PRELOADEDNEIGHBOURHOODS_H_
#error File preloadedNeighbourhoods.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
PreloadedNeigborhoods<PixelType, MYSIZE>::PreloadedNeigborhoods()
	: _loading(0), _loaded(0), _saving(0)
{
#ifdef FOR_CELL
	for(uint32 i=0; i<MYSIZE; i++)
		m_buf[i]._loadingCtx = &_loadingCtx;
#endif
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>
::SetImageProps(TImageProps *properties)
{
	_imageProps = properties;
	uint32 i;
	
	for(i=0; i<MYSIZE; i++)
	{
		m_buf[i].SetImageProperties(properties);
	}
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>
::Load(const SparseFieldLevelSetNode &node)
{
#ifdef FOR_CELL
	if(_loadingCtx.tagMask > 0)
		WaitForLoading();
#endif
	
	// change pointers
	_loading++;
	_loading = _loading % MYSIZE;
	
	_loadedNodeNexts[_loading] = node.Next;
	m_buf[_loading].SetPosition(node.m_Value);
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType, uint16 MYSIZE>
typename PreloadedNeigborhoods<PixelType, MYSIZE>::TNeigborhood *
PreloadedNeigborhoods<PixelType, MYSIZE>
::GetLoaded()
{
	_loaded++;
	_loaded = _loaded % MYSIZE;
	return &m_buf[_loaded];
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::SaveCurrItem()
{	
	// do nothing if we are on PC
#ifdef FOR_CELL
	if(_savingCtx.tagMask > 0)
		WaitForSaving();
	
	m_buf[_loaded].SaveChanges(&_savingCtx);
	
	uint32 tagIter = 0;
	// issue the lists	
	for(uint32 i=0; i<SAVE_DMA_LIST_CNT; i++)
	{
		if(_savingCtx._dmaListIter[i])
		{
			if(tagIter == 3)
			{
				uint32 i=10; i++;
				D_PRINT("Sem Neeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee!!!!!!!!!!!!!!!!!!!!");
			}
			DMAGate::PutList(
						_imageProps->imageData.Get64(), 
						_savingCtx.tmpBuf, 
						_savingCtx.dma_list[i], 
						_savingCtx._dmaListIter[i],
						_savingCtx.tags[tagIter]);
			_savingCtx.tagMask |= (1 << _savingCtx.tags[tagIter]);
			tagIter++;
		}
	}
	
	// change pointers
	_saving = _loaded;
#endif
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::Reset()
{
	//_loading = _loaded = _saving = 0;
	_loaded = _loading;
	memset(_loadedNodeNexts, 0, MYSIZE * sizeof(Address));
}

///////////////////////////////////////////////////////////////////////////////
// CELL only part
///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL

template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::ReserveTags()
{
	// reserve necessary tags
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		_loadingCtx.tags[i] = DMAGate::GetTag();
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_GET:NeighborhoodCell:%d\n", _loadingCtx.tags[i]);
#endif
	}
	
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		_savingCtx.tags[i] = DMAGate::GetTag();
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_GET:NeighborhoodCell:%d\n", _savingCtx.tags[i]);
#endif
	}
}
///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::ReturnTags()
{
	// return loading tags
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_RET:PreloadedNeigborhoods:%d\n", _loadingCtx.tags[i]);
#endif
		DMAGate::ReturnTag(_loadingCtx.tags[i]);
	}
	
	// return saving tags
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
#ifdef TAG_RETURN_DEBUG
		D_PRINT("TAG_RET:PreloadedNeigborhoods:%d\n", _savingCtx.tags[i]);
#endif
		DMAGate::ReturnTag(_savingCtx.tags[i]);
	}
}


///////////////////////////////////////////////////////////////////////////////
//template<typename PixelType, uint16 MYSIZE>
//void
//PreloadedNeigborhoods<PixelType, MYSIZE>::Fini()
//{
//	WaitForLoading();
//	WaitForSaving();
//}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::WaitForLoading()
{
	// wait until loading is finished
	mfc_write_tag_mask(_loadingCtx.tagMask);
	mfc_read_tag_status_all();
	
	_loadingCtx.tagMask = 0;
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::WaitForSaving()
{
	// wait until loading is finished
	mfc_write_tag_mask(_savingCtx.tagMask);
	mfc_read_tag_status_all();
	
	_savingCtx.tagMask = 0;
}
#endif
///////////////////////////////////////////////////////////////////////////////

#endif
