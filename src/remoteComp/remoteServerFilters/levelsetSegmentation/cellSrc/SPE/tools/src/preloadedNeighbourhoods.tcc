#ifndef PRELOADEDNEIGHBOURHOODS_H_
#error File preloadedNeighbourhoods.tcc cannot be included directly!
#else

namespace M4D {
namespace Cell {

///////////////////////////////////////////////////////////////////////////////
template<typename TNeigborhood, uint16 MYSIZE>
PreloadedNeigborhoods<TNeigborhood, MYSIZE>::PreloadedNeigborhoods()
{	
}

///////////////////////////////////////////////////////////////////////////////
template<typename TNeigborhood, uint16 MYSIZE>
void
PreloadedNeigborhoods<TNeigborhood, MYSIZE>
::SetImageProps(TImageProps *properties)
{
	uint32 i;
	
	for(i=0; i<MYSIZE; i++)
	{
		m_buf[i].SetImageProperties(properties);
	}
}

///////////////////////////////////////////////////////////////////////////////
template<typename TNeigborhood, uint16 MYSIZE>
void
PreloadedNeigborhoods<TNeigborhood, MYSIZE>
::Load(const TIndex &pos)
{
	// use the first
	m_buf[0].SetPosition(pos);
}

///////////////////////////////////////////////////////////////////////////////
template<typename TNeigborhood, uint16 MYSIZE>
TNeigborhood *
PreloadedNeigborhoods<TNeigborhood, MYSIZE>
::GetLoaded()
{
	return &m_buf[0];
}

///////////////////////////////////////////////////////////////////////////////
template<typename TNeigborhood, uint16 MYSIZE>
void
PreloadedNeigborhoods<TNeigborhood, MYSIZE>::SaveCurrItem()
{
	// do nothing if we are on PC
}

///////////////////////////////////////////////////////////////////////////////

}
}

#endif
