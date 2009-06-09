#ifndef APPLYUPDATECALCULATOR_H_
#define APPLYUPDATECALCULATOR_H_

#include "layerValsPropagator.h"
#include "../tools/objectStoreCell.h"

namespace M4D {
namespace Cell {

class ApplyUpdateSPE : public LayerValuesPropagator
{
public:
	ApplyUpdateSPE(SharedResources *shaRes);
	~ApplyUpdateSPE();
	
	typedef float32 ValueType;
	
	ValueType ApplyUpdate(TimeStepType dt);
	
	typedef GETRemoteArrayCell<TPixelValue, REMOTEARRAY_BUF_SIZE> TUpdateBufferArray;	
	typedef SparseFieldLayer<SparseFieldLevelSetNode> MyLayerType;
	
#define LocalNodeStoreSize 128
	typedef FixedObjectStoreCell<SparseFieldLevelSetNode, LocalNodeStoreSize> TObjectStore; 
	
	void UpdateActiveLayerValues(TimeStepType dt,
			MyLayerType *UpList, MyLayerType *DownList,
			uint32 &counter, ValueType &rms_change_accumulator);
		
			
	void ProcessOutsideList(MyLayerType *OutsideList, StatusType ChangeToStatus);//, TStatusNeighbIterator &statIter);
	void ProcessStatusList(MyLayerType *InputList, MyLayerType *OutputList,
            StatusType ChangeToStatus, StatusType SearchForStatus);//, TStatusNeighbIterator &statusIt);
private:
	
	void
	ProcessStatusLists(MyLayerType *UpLists, MyLayerType *DownLists);
	
	
	inline ValueType CalculateUpdateValue(
		    const TimeStepType &dt,
		    const ValueType &value,
		    const ValueType &change)
		    {
			ValueType val = (value + dt * change); 
			return val;
			}
	
//	SparseFieldLevelSetNode *BorrowFromLocalNodeStore()
//	{
//		return this->m_layerGate.m_LayerNodeStore->Borrow();
//	}
	LayerValuesPropagator::TStatusNeighbPreloadeder m_statusUpdatePreloader;
	
	CalculateChangeAndUpdActiveLayerConf *m_stepConfig;
	

	SparseFieldLevelSetNode m_localNodeStoreBuffer[LocalNodeStoreSize];
	TObjectStore m_localNodeStore;
	TUpdateBufferArray m_updateValuesIt;
	
	uint32 m_ElapsedIterations;
	
	// supp members
	SparseFieldLevelSetNode *_loaded;
};

}
}
#endif /*APPLYUPDATECALCULATOR_H_*/
