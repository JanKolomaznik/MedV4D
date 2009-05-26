#ifndef NEIGHBORHOODCELL_H_
#error File neighborhoodCell.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
NeighborhoodCell<PixelType>::NeighborhoodCell()
		: m_imageProps(0)
{
	// set size
	for (unsigned int i=0; i<DIM; ++i)
		{ m_radiusSize[i] = SIZEIN1DIM; }
	
	// count size of buffer in linear manner
	m_size = NEIGHBOURHOOD_SIZE;
	
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

//template<typename PixelType>
//void
//NeighborhoodCell<PixelType>
//::LoadData(PixelType *src, PixelType *dest, size_t size)
//{
//	// copy the memory
//	memcpy((void*)dest, (void*)src, size);
//}

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

//template<typename PixelType>
//void
//NeighborhoodCell<PixelType>
//::ComputeAlignStrides()
//{
//	for(uint i=0; i<DIM-2; i++)
//	{
//		alignStrideTable_[i] = 
//			(m_imageStrides[i+1] - ((SIZEIN1DIM * m_imageStrides[i]) % 16)) % 16;
//	}
//}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
template<typename PixelType>
void
NeighborhoodCell<PixelType>
::PutIntoList(uint64 address, uint32 size)
{
	// align within quadword
	uint32 _alignIter = (uint32) (address & 0xF) / sizeof(PixelType);
	dma_list[_alignIter][_dmaListIter[_alignIter]].notify = 0;
	dma_list[_alignIter][_dmaListIter[_alignIter]].eal = address;
	dma_list[_alignIter][_dmaListIter[_alignIter]].size = size;
	
	// setup translation table
	uint32 beginInBuf = _alignIter + (_dmaListIter[_alignIter] * 16);
	for(uint32 i=0; i<size; i++)
		traslationTable_[transIdxIter_++] = beginInBuf + i;
	
	_dmaListIter[_alignIter]++;
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
			PutIntoList(begin.Get64(), sizeof(PixelType) * (SIZEIN1DIM-1));
#else
			DMAGate::Get(begin, dest, (SIZEIN1DIM-1) * sizeof(PixelType) );
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
			PutIntoList(begin.Get64(), sizeof(PixelType) * (SIZEIN1DIM-1));			
#else
			DMAGate::Get(begin, dest, (SIZEIN1DIM-1) * sizeof(PixelType) );
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
			PutIntoList(begin.Get64(), sizeof(PixelType) * (SIZEIN1DIM-1));
			PutIntoList(begin.Get64() + (sizeof(PixelType) * (SIZEIN1DIM-1)), sizeof(PixelType));
#else
			DMAGate::Get(begin, dest, SIZEIN1DIM * sizeof(PixelType) );
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
	PixelType *begin = (PixelType *) ComputeImageDataPointer(m_currIndex).Get64();
	*begin = val;
	// change the buffer as well
	m_buf[static_cast<uint32>(m_size/2)] = val;
}

///////////////////////////////////////////////////////////////////////////////

template<typename PixelType>
void
NeighborhoodCell<PixelType>::SetPixel(PixelType val, TOffset pos)
{
	TIndex i = m_currIndex + pos;
	if(IsWithinImage(i))
	{
		PixelType *begin = (PixelType *) ComputeImageDataPointer(i).Get64();
		*begin = val;
	}
	// change the buffer as well !!!!!!!!
	//std::cout << "setting" << i << "=" << val << std::endl;
	m_buf[GetNeighborhoodIndex(pos)] = val;
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
	memset((void*)m_buf, DEFAULT_VAL, m_size * sizeof(PixelType));
	transIdxIter_ = 0;
	
#ifdef FOR_CELL
	memset((void*)_dmaListIter, 0, LIST_SET_NUM * sizeof(uint32));
	memset((void*)traslationTable_, 0xFF, m_size * sizeof(int32));
#endif	
	
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
	uint32 tags[LIST_SET_NUM];
	uint32 mask = 0;
	for(uint32 i=0; i<LIST_SET_NUM; i++)
	{
		if(_dmaListIter[i])
		{
			tags[i] = DMAGate::GetList(
						m_imageProps->imageData.Get64(), 
						m_buf, dma_list[i], _dmaListIter[i]);
			mask |= (1 << tags[i]);
		}
	}
	
	mfc_write_tag_mask(mask);
	mfc_read_tag_status_all();
	
	// TODO return tags into gate
	
//	for(uint i=0; i<BUFFER_SIZE; i++)
//		D_PRINT("%u = %u\n", i, m_buf[i]);
#endif
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
				data = ComputeImageDataPointer(ind);
				s << "[" << ind[0] << "," << ind[1] << "," << ind[2] << "]"  << "= " << ((int32)*data) << std::endl;
			}
		}
	}
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

}
}
#endif
