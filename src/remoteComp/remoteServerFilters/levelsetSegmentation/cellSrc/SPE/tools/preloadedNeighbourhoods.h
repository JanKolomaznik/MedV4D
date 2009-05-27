#ifndef PRELOADEDNEIGHBOURHOODS_H_
#define PRELOADEDNEIGHBOURHOODS_H_

#include "neighborhoodCell.h"

namespace M4D
{
namespace Cell
{

template<typename PixelType, uint16 MYSIZE>
class PreloadedNeigborhoods
{
public:
	typedef NeighborhoodCell<PixelType> TNeigborhood;
	typedef typename TNeigborhood::TImageProps TImageProps;
	
	PreloadedNeigborhoods();
	
	void SetImageProps(TImageProps *properties);
	
	void Load(const TIndex &pos);
	TNeigborhood *GetLoaded();
	void SaveCurrItem();
private:
	TNeigborhood m_buf[MYSIZE];
	
	typename TNeigborhood::LoadingCtx _loadingCtx;
	typename TNeigborhood::SavingCtx _savingCtx;
	
	void WaitForLoading();
	void WaitForSaving();
	
	bool _loadingInProgress;
	bool _savingInProgress;
	uint8 _loading, _loaded, _saving;
	
	TImageProps *_imageProps;
};

#include "src/preloadedNeighbourhoods.tcc"

}
}

#endif /*PRELOADEDNEIGHBOURHOODS_H_*/
