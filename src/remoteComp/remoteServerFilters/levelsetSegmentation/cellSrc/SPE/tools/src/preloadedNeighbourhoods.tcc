#ifndef PRELOADEDNEIGHBOURHOODS_H_
#error File preloadedNeighbourhoods.tcc cannot be included directly!
#else

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
PreloadedNeigborhoods<PixelType, MYSIZE>::PreloadedNeigborhoods()
	: _loading(0), _loaded(0), _saving(0), _loadingInProgress(false)
	, _savingInProgress(false)
{
	for(uint i=0; i<MYSIZE; i++)
		m_buf[i]._loadingCtx = &_loadingCtx;
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
::Load(const TIndex &pos)
{
	if(_loadingInProgress)
		WaitForLoading();
	// change pointers
	_loaded = _loading;
	_loading++;
	_loading = _loading % MYSIZE;
	m_buf[_loading].SetPosition(pos);
	_loadingInProgress = true;
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::WaitForLoading()
{
	// wait until loading is finished
	mfc_write_tag_mask(_loadingCtx.tagMask);
	mfc_read_tag_status_all();
	
	// return tags
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		if(_loadingCtx.tags[i])
			DMAGate::ReturnTag(_loadingCtx.tags[i]);
	}
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::WaitForSaving()
{
	// wait until loading is finished
	mfc_write_tag_mask(_savingCtx.tagMask);
	mfc_read_tag_status_all();
	
	// return tags
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		if(_savingCtx.tags[i])
			DMAGate::ReturnTag(_savingCtx.tags[i]);
	}
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
typename PreloadedNeigborhoods<PixelType, MYSIZE>::TNeigborhood *
PreloadedNeigborhoods<PixelType, MYSIZE>
::GetLoaded()
{
	return &m_buf[_loaded];
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 MYSIZE>
void
PreloadedNeigborhoods<PixelType, MYSIZE>::SaveCurrItem()
{	
	// do nothing if we are on PC
#ifdef FOR_CELL
	if(_savingInProgress)
			WaitForSaving();
	
	m_buf[_loaded].SaveChanges(&_savingCtx);
	
	// issue the lists
	_savingCtx.tagMask = 0;
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		if(_savingCtx._dmaListIter[i])
		{
			_savingCtx.tags[i] = DMAGate::PutList(
						_imageProps->imageData.Get64(), 
						_savingCtx.tmpBuf, 
						_savingCtx.dma_list[i], 
						_savingCtx._dmaListIter[i]);
			_savingCtx.tagMask |= (1 << _savingCtx.tags[i]);
		}
	}
	
	if(_savingCtx.tagMask > 0)
		_savingInProgress = true;
	else
		_savingInProgress = false;
	
	// change pointers
	_saving = _loaded;
#endif
}

///////////////////////////////////////////////////////////////////////////////

#endif
