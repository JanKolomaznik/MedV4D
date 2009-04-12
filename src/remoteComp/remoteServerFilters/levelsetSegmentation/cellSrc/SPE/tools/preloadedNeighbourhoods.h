#ifndef PRELOADEDNEIGHBOURHOODS_H_
#define PRELOADEDNEIGHBOURHOODS_H_

namespace M4D
{
namespace Cell
{

template<typename TNeigborhood, uint16 MYSIZE>
class PreloadedNeigborhoods
{
public:
	typedef typename TNeigborhood::TImageProps TImageProps;
	
	PreloadedNeigborhoods();
	
	void SetImageProps(TImageProps *properties);
	
	void Load(const TIndex &pos);
	TNeigborhood *GetLoaded();
	void SaveCurrItem();
private:
	TNeigborhood m_buf[MYSIZE];
};

}
}

#include "src/preloadedNeighbourhoods.tcc"

#endif /*PRELOADEDNEIGHBOURHOODS_H_*/
