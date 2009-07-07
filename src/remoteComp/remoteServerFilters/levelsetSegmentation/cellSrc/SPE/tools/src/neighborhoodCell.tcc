#ifndef NEIGHBORHOODCELL_H_
#error File neighborhoodCell.tcc cannot be included directly!
#else


///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
NeighborhoodCell<PixelType>::NeighborhoodCell()
		: m_imageProps(0), _whichDMAList(0), numOfLoadings(0), numOfSavings(0)
{
	// set size
	for (unsigned int i=0; i<DIM; ++i)
		{ m_radiusSize[i] = SIZEIN1DIM; }
	
	// count size of buffer in linear manner
	m_size = NEIGHBOURHOOD_SIZE;
	
	// zero the buff
	memset((void*)m_buf, 0, BUFFER_SIZE * sizeof(PixelType));
	
	ComputeStridesFromSize<TSize, TStrides>(m_radiusSize, m_radiusStrides);
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
Address
NeighborhoodCell<PixelType>
::ComputeImageDataPointer(const TIndex &pos)
{
	Address pointer = m_imageProps->imageData;
	for(uint8 i=0; i<DIM; i++)
	{
		pointer += pos[i] * m_imageStrides[i] * sizeof(PixelType);
	}
	return pointer;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
bool
NeighborhoodCell<PixelType>
::IsWithinImage(const TIndex &pos)
{
	for(uint32 i=0; i<DIM; i++)
	{
		if((pos[i] < m_imageProps->region.offset[i]) 
				|| (pos[i] >= (m_imageProps->region.offset[i] + (int32)m_imageProps->region.size[i]) ) )
			return false;
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
bool
NeighborhoodCell<PixelType>
::IsWithinImageContinuos( const TContinuousIndexDouble & index ) const
{
	  if( (index[0] >= m_imageProps->region.offset[0] )
		&& (index[0] <= (m_imageProps->region.offset[0] + m_imageProps->region.size[0] -1))
		
	    && (index[1] >= m_imageProps->region.offset[1] )
	    && (index[1] <= (m_imageProps->region.offset[1] + m_imageProps->region.size[1] -1))
	    
	    && (index[2] >= m_imageProps->region.offset[2] )
	    && (index[2] <= (m_imageProps->region.offset[2] + m_imageProps->region.size[2] -1)) 
	    )
		  return true;
	  else
		  return false;
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
bool
NeighborhoodCell<PixelType>::IsWithinNeigbourhood(const TIndex &pos)
{
		if( 
				(pos[0] >= (m_currIndex[0] - RADIUS))
		&& 		(pos[0] <= (m_currIndex[0] + RADIUS)) 
		&&		(pos[1] >= (m_currIndex[1] - RADIUS))
		&& 		(pos[1] <= (m_currIndex[1] + RADIUS))
		&&		(pos[2] >= (m_currIndex[2] - RADIUS))
		&& 		(pos[2] <= (m_currIndex[2] + RADIUS))
		)
		{
			return true;
		}
		else			
			return false;
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
template<typename PixelType>
void
NeighborhoodCell<PixelType>
::PutIntoList(uint64 address, uint32 size)
{
	// align within quadword
	uint32 _alignIter = (uint32) (address & 0xF) / sizeof(PixelType);
	// use approp list based on _alignIter
	_loadingCtx->dma_list[_whichDMAList][_loadingCtx->_dmaListIter[_whichDMAList]].notify = 0;
	_loadingCtx->dma_list[_whichDMAList][_loadingCtx->_dmaListIter[_whichDMAList]].eal = address;
	_loadingCtx->dma_list[_whichDMAList][_loadingCtx->_dmaListIter[_whichDMAList]].size = size * sizeof(PixelType);
	
	// setup translation table
	uint32 beginInBuf = 
		_alignIter + (_loadingCtx->_dmaListIter[_whichDMAList] * 16 / sizeof(PixelType));
	for(uint32 i=0; i<size; i++)
		traslationTable_[transIdxIter_++] = beginInBuf + i;
	
	_loadingCtx->_dmaListIter[_whichDMAList]++;	// increase curr list count
	_whichDMAList = ! _whichDMAList;	// swap the lists
}
#endif
///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
void
NeighborhoodCell<PixelType>
::LoadSlice(TIndex posm, uint8 dim, PixelType *dest)
{
	posm[dim] -= RADIUS;
	if(dim == 0)
	{
		Address begin;
		if(posm[0] < 0)
		{
			// begin of array is -1, so ...
			// load the array from 0 ...
			posm[0] = 0;
			transIdxIter_++;	// skip the out of image elem
			begin = ComputeImageDataPointer(posm);
			// ... and only SIZEIN1DIM-1 elems
#ifdef FOR_CELL
			PutIntoList(begin.Get64(), SIZEIN1DIM-1);
			_whichDMAList = ! _whichDMAList;
#else
			DMAGate::Get(begin, dest, sizeof(PixelType) * (SIZEIN1DIM-1));
			for(uint32 i=0; i<(SIZEIN1DIM-1); i++)
			{
				traslationTable_[transIdxIter_] = transIdxIter_;
				transIdxIter_++;
			}
#endif
		}
		else if((posm[0]+m_radiusSize[dim]) >= m_imageProps->region.size[0])
		{
			// end of array is out of omage, so ...
			begin = ComputeImageDataPointer(posm);
			// ... load only SIZEIN1DIM-1 elems			
#ifdef FOR_CELL
			PutIntoList(begin.Get64(), SIZEIN1DIM-1);
			_whichDMAList = ! _whichDMAList;
#else
			DMAGate::Get(begin, dest, (SIZEIN1DIM-1) * sizeof(PixelType));
			for(uint32 i=0; i<(SIZEIN1DIM-1); i++)
			{
				traslationTable_[transIdxIter_] = transIdxIter_;
				transIdxIter_++;
			}
#endif
			transIdxIter_++;	// skip the out of image elem
		}
		else
		{
			begin = ComputeImageDataPointer(posm);
			// load whole SIZEIN1DIM-1 elems of array
#ifdef FOR_CELL
#define BIT_CNT_NEED_TO_BE_EQUAL_FOR_TRASFER_SIZE(x) ((sizeof(PixelType)*(x))-1)
			if((begin.Get64() & BIT_CNT_NEED_TO_BE_EQUAL_FOR_TRASFER_SIZE(2)) == 0)
			{
				PutIntoList(begin.Get64(), SIZEIN1DIM-1);
				PutIntoList(begin.Get64() + (sizeof(PixelType) * (SIZEIN1DIM-1)), 1);
			}
			else
			{
				PutIntoList(begin.Get64(), 1);	// put into list in reverse order
				PutIntoList(begin.Get64() + sizeof(PixelType), SIZEIN1DIM-1);
			}
#else
			DMAGate::Get(begin, dest, SIZEIN1DIM * sizeof(PixelType));
			for(uint32 i=0; i<SIZEIN1DIM; i++)
			{
				traslationTable_[transIdxIter_] = transIdxIter_;
				transIdxIter_++;
			}
#endif
		}
	}
	else
	{
		TIndex iteratingIndex(posm);
		for(uint32 i=0; i<m_radiusSize[dim]; i++)
		{
			if(IsWithinImage(iteratingIndex))
			{
				LoadSlice(iteratingIndex, dim-1, dest + (i * m_radiusStrides[dim]));				
			}
			else
			{
				transIdxIter_ += SIZEIN1DIM;  // one slice's row
			}
			iteratingIndex[dim] += 1;	// move in iteration  direction
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>::SetCenterPixel(PixelType val)
{	
	//D_PRINT("SET center: " << val);
	_dirtyElems |= (1 << (m_size/2));	// set dirty flag
	// change the buffer as well
	m_buf[traslationTable_[static_cast<uint32>(m_size/2)]] = val;	
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>::SetPixel(PixelType val, TOffset pos)
{
	TIndex i = m_currIndex + pos;
	if(IsWithinImage(i))
	{
		_dirtyElems |= (1 << GetNeighborhoodIndex(pos));	// set dirty flag
		// and change the buffer as well
		m_buf[traslationTable_[GetNeighborhoodIndex(pos)]] = val;
	}
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>
	::SetPosition(const TIndex &pos)
{
	m_currIndex = pos;
	
#define DEFAULT_VAL 0
	// fill the buff
	memset((void*)m_buf, DEFAULT_VAL, BUFFER_SIZE * sizeof(PixelType));
	transIdxIter_ = 0;
	_dirtyElems = 0;
	
#ifdef FOR_CELL
	memset((void*)_loadingCtx->_dmaListIter, 0, LIST_SET_NUM * sizeof(TDmaListIter));
#endif
	memset((void*)traslationTable_, 0xFF, NEIGHBOURHOOD_SIZE * sizeof(int32));
	
	TIndex iteratingIndex(pos);
	iteratingIndex[DIM-1] -= RADIUS;
	for(uint32 i=0; i<m_radiusSize[DIM-1]; i++)
	{
		if(IsWithinImage(iteratingIndex))
		{
			LoadSlice(iteratingIndex, DIM-2, m_buf + (i * m_radiusStrides[DIM-1]));
		}
		else
		{
			transIdxIter_ += NEIGHBOURHOOD_SLICE_SIZE;
		}
		iteratingIndex[DIM-1] += 1;	// move in iteration  direction
	}
	
#ifdef FOR_CELL	
	// issue the lists
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		if(_loadingCtx->_dmaListIter[i])
		{
			DMAGate::GetList(
						m_imageProps->imageData.Get64(), 
						m_buf, 
						_loadingCtx->dma_list[i], 
						_loadingCtx->_dmaListIter[i],
						_loadingCtx->tags[i]);
			_loadingCtx->tagMask |= (1 << _loadingCtx->tags[i]);
		}
	}
	
	numOfLoadings++;
#endif
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
template<typename PixelType>
void
NeighborhoodCell<PixelType>::SaveChanges(SavingCtx *ctx)
{
	register uint32 cnt = 0;
	register uint32 dirtyMask = _dirtyElems;
	
	// clear list iters
	memset((void*)ctx->_dmaListIter, 0, SAVE_DMA_LIST_CNT * sizeof(TDmaListIter));
	
	uint64 address;
	while((cnt < 27))
	{
		if(dirtyMask & 0x1)
		{
			numOfSavings++;
			

			address = 
				ComputeImageDataPointer(m_currIndex + OffsetFromPos(cnt)).Get64();
			uint32 _alignIter = (uint32) (address & 0xF) / sizeof(PixelType);
			
			//printf("SaveChanges: align=%d\n", _alignIter);
				
			ctx->dma_list[_alignIter][ctx->_dmaListIter[_alignIter]].notify = 0;
			ctx->dma_list[_alignIter][ctx->_dmaListIter[_alignIter]].eal = address;
			ctx->dma_list[_alignIter][ctx->_dmaListIter[_alignIter]].size = sizeof(PixelType);
			
			// move to the front of buff because DMA list transfer continuous 
			// local store array part
			uint32 beginInBuf = 
					_alignIter + (ctx->_dmaListIter[_alignIter] * 16 / sizeof(PixelType));
			ctx->tmpBuf[beginInBuf] = m_buf[traslationTable_[cnt]];
			
			ctx->_dmaListIter[_alignIter]++;
		}
		
		dirtyMask >>= 1;	// shift right
		cnt++;
	}
}
#else
template<typename PixelType>
void
NeighborhoodCell<PixelType>::SaveChanges()
{
	register uint32 cnt = 0;
	register uint32 dirtyMask = _dirtyElems;
	
	Address address;
	while(dirtyMask && (cnt < 27))
	{
		if(dirtyMask & 0x1)
		{
			numOfSavings++;
			

			address = 
				ComputeImageDataPointer(m_currIndex + OffsetFromPos(cnt));
			//D_PRINT("Eventually:" << (void *) address);
			DMAGate::Put(&m_buf[cnt], address, sizeof(PixelType));
		}
		
		dirtyMask >>= 1;	// shift right
		cnt++;
	}
}
#endif

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
void
NeighborhoodCell<PixelType>::PropagateChangesWithinSavedItem(Self& saved)
{
	uint32 cnt = 0;
	uint32 savedDirty = saved._dirtyElems;
	
	Address address;
	while(savedDirty && (cnt < NEIGHBOURHOOD_SIZE))
	{
		if(savedDirty & 0x1)
		{
			TIndex savedsPos = saved.m_currIndex + OffsetFromPos(cnt);
			// if savedsPos overlap with our neighb bbox
			if(IsWithinNeigbourhood(savedsPos))
			{
				TOffset ourPos = savedsPos - m_currIndex;
				uint32 ourIndex = GetNeighborhoodIndex(ourPos);
				m_buf[traslationTable_[ourIndex]] = saved.GetPixel(cnt);
			}			
		}
		
		savedDirty >>= 1;	// shift right
		cnt++;
	}
}
///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
TOffset
NeighborhoodCell<PixelType>::OffsetFromPos(uint32 pos)
{
	TOffset idx;
	uint32 acc = pos;
	for(int32 i=DIM-1; i>=0; i--)
	{
		idx[i] = (acc / m_radiusStrides[i]) - 1;
		acc %= m_radiusStrides[i];
	}
	return idx;
}

///////////////////////////////////////////////////////////////////////////////
template<typename PixelType>
uint32 
NeighborhoodCell<PixelType>::GetNeighborhoodIndex(const TOffset &o) const
{
  uint32 idx = (m_size/2);
  for (unsigned i = 0; i < DIM; ++i)
    {      idx+=o[i] * static_cast<long>(m_radiusStrides[i]);    }
  return idx;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>
::HowMuchCrossesBoundary(TOffset &howMuch)
{
	TIndex pos = m_currIndex + howMuch;
	for(uint32 i=0; i<DIM; i++)
	{
		if(pos[i] < m_imageProps->region.offset[i])
			howMuch[i] = m_imageProps->region.offset[i] - pos[i];
		else if(pos[i] >= (m_imageProps->region.offset[i] + (int32)m_imageProps->region.size[i]) )
			howMuch[i] = (m_imageProps->region.offset[i] + (int32)m_imageProps->region.size[i] - 1) - pos[i];
		else
			howMuch[i] = 0;
	}
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
std::ostream & operator<<(std::ostream &stream, NeighborhoodCell<PixelType> &n)
{
	stream << "m_currIndex: " << n.m_currIndex << std::endl;
	
	TOffset iteratingIndex;
	for(uint32 i=0; i<DIM; i++)
	{
		iteratingIndex[i] = -RADIUS;
	}
	TOffset begin(iteratingIndex);
	PixelType val;
	
	for(uint32 i=0; i<SIZEIN1DIM; i++)
	{
		iteratingIndex[1] = begin[1];
		for(uint32 j=0; j<SIZEIN1DIM; j++)
		{
			iteratingIndex[0] = begin[0];
			for(uint32 k=0; k<SIZEIN1DIM; k++)
			{
				val = n.GetPixel(n.GetNeighborhoodIndex(iteratingIndex));
				stream << (iteratingIndex) << "=" << (int32) val << std::endl;
				iteratingIndex[0] += 1;
			}
			iteratingIndex[1] += 1;
		}
		iteratingIndex[2] += 1;
	}
	return stream;
}

///////////////////////////////////////////////////////////////////////////////


template<typename PixelType>
void
NeighborhoodCell<PixelType>::Print()
{
#ifdef FOR_CELL
	D_PRINT("m_currIndex: [%d, %d, %d]\n", m_currIndex[0], m_currIndex[1], m_currIndex[2]);
#else
	D_PRINT("m_currIndex: [" << m_currIndex[0] << ", " << m_currIndex[1] <<
			", " << m_currIndex[2] << "]");
#endif
	
	int8 pos;
	
	for(uint32 i=0; i<NEIGHBOURHOOD_SIZE; i++)
	{
		pos = traslationTable_[i];
		if(pos != -1)
		{
#ifdef FOR_CELL
		D_PRINT("%f,", (float32) m_buf[pos]);
#else
		D_PRINT_NOENDL((float32) m_buf[pos] << ",");
#endif
		}
	}
	
#ifdef FOR_CELL
	D_PRINT("\n");
#else
	D_PRINT("");
#endif
}
///////////////////////////////////////////////////////////////////////////////
#ifndef FOR_CELL
template<typename PixelType>
void
NeighborhoodCell<PixelType>::PrintImage(std::ostream &s)
{
	s << "size: " << m_imageProps->region.size[0] << "," << m_imageProps->region.size[1] << "," << m_imageProps->region.size[2] << std::endl;
	PixelType *data;
	TIndex ind;
	
	for(uint32 i=0; i<m_imageProps->region.size[0]; i++)
	{
		for(uint32 j=0; j<m_imageProps->region.size[1]; j++)
		{
			for(uint32 k=0; k<m_imageProps->region.size[2]; k++)
			{
				ind[0] = i; ind[1] = j; ind[2] = k;
				data = (PixelType *)ComputeImageDataPointer(ind).Get64();
				s << "[" << ind[0] << "," << ind[1] << "," << ind[2] << "]"  << "= " << (float32)(*data) << std::endl;
			}
		}
	}
}
template<typename PixelType>
void
NeighborhoodCell<PixelType>::PrintImageToFile(const char *fileName)
{
	std::ofstream s(fileName);
	PrintImage(s);
	s.close();
}
#endif //FOR_CELL
///////////////////////////////////////////////////////////////////////////////

#endif
