#ifndef APPLYUPDATECALCULATOR_H_
#define APPLYUPDATECALCULATOR_H_

#include "layerValsPropagator.h"

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
	
	ValueType UpdateActiveLayerValues(TimeStepType dt,
			MyLayerType *UpList, MyLayerType *DownList);
		            //, TValueNeighbIterator &outIt, TStatusNeighbIterator &statusIt);
		
			
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
	
	SparseFieldLevelSetNode *BorrowFromLocalNodeStore()
	{
		return m_LayerNodeStore->Borrow();
	}
	

	
	
	
	uint32 m_ElapsedIterations;
};

}
}
#endif /*APPLYUPDATECALCULATOR_H_*/
