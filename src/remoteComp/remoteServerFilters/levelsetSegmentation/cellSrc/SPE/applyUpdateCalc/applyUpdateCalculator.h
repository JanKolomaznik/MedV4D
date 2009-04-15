#ifndef APPLYUPDATECALCULATOR_H_
#define APPLYUPDATECALCULATOR_H_

#include "layerValsPropagator.h"

#include "../tools/objectStoreCell.h"

namespace M4D {
namespace Cell {

class ApplyUpdateSPE : public LayerValuesPropagator
{
public:
	ApplyUpdateSPE();
	~ApplyUpdateSPE();
	
	typedef float32 ValueType;
	
	ValueType ApplyUpdate(TimeStepType dt);		
	
	typedef GETRemoteArrayCell<TPixelValue, 8> TUpdateBufferArray;	
	typedef M4D::Cell::SparseFieldLayer<SparseFieldLevelSetNode> MyLayerType;
	typedef M4D::Cell::ObjectStoreCell<SparseFieldLevelSetNode, 2048> TObjectStore; 
	
	void UpdateActiveLayerValues(TimeStepType dt,
			MyLayerType *UpList, MyLayerType *DownList,
			uint32 &counter, ValueType &rms_change_accumulator);
		
	CalculateChangeAndUpdActiveLayerConf *m_stepConfig;
			
	void ProcessOutsideList(MyLayerType *OutsideList, StatusType ChangeToStatus);//, TStatusNeighbIterator &statIter);
	void ProcessStatusList(MyLayerType *InputList, MyLayerType *OutputList,
            StatusType ChangeToStatus, StatusType SearchForStatus);//, TStatusNeighbIterator &statusIt);
private:
	
	void
	ProcessStatusLists(MyLayerType *UpLists, MyLayerType *DownLists);
	
	
	ValueType CalculateUpdateValue(
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
	

	TObjectStore m_localNodeStore;
	TUpdateBufferArray m_updateValuesIt;
	
	uint32 m_ElapsedIterations;
};

}
}
#endif /*APPLYUPDATECALCULATOR_H_*/
