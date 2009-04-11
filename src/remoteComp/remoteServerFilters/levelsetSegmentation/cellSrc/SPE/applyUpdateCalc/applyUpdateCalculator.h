#ifndef APPLYUPDATECALCULATOR_H_
#define APPLYUPDATECALCULATOR_H_

#include "common/Types.h"
#include "../configStructures.h"
// tools
#include "../tools/neighbourhoodIterator.h"
#include "../tools/cellRemoteArray.h"
#include "../tools/cellLinkedChainIterator.h"

#include "../../supportClasses.h"
#include "../../commonConsts.h"

#include "../tools/sparesFieldLayer.h"

// to remove
#include "itkSparseFieldLayer.h"
#include "itkObjectStore.h"

namespace M4D {
namespace Cell {

class ApplyUpdateSPE : public Consts
{
public:
	ApplyUpdateSPE();
	~ApplyUpdateSPE();
	
	typedef float32 ValueType;
	
	ValueType ApplyUpdate(TimeStepType dt);
	
	void PropagateAllLayerValues();
	
	ApplyUpdateConf conf;
	RunConfiguration *commonConf;
	
	void SetCommonConfiguration(RunConfiguration *c) { commonConf = c; }
	
	typedef NeighbourIteratorCell<TPixelValue> TValueNeighbIterator;
	typedef NeighbourIteratorCell<StatusType> TStatusNeighbIterator;
	
	typedef GETRemoteArrayCell<TPixelValue, 8> TUpdateBufferArray;
	
	typedef M4D::Cell::SparseFieldLayer<SparseFieldLevelSetNode> MyLayerType;
	
	//to remove
	typedef itk::SparseFieldLayer<SparseFieldLevelSetNode> LayerType;
	typedef LayerType::Pointer LayerPointerType;
	
	LayerType **m_Layers;
	typedef itk::ObjectStore<SparseFieldLevelSetNode> LayerNodeStorageType;
	LayerNodeStorageType *m_LayerNodeStore;
	
//	ValueType UpdateActiveLayerValues(TimeStepType dt,
//	            LayerType *UpList, LayerType *DownList);
//	            //, TValueNeighbIterator &outIt, TStatusNeighbIterator &statusIt);
//	void PropagateLayerValues(StatusType from, StatusType to,
//		                       StatusType promote, uint32 InOrOut);
//		
//		void ProcessOutsideList(LayerType *OutsideList, StatusType ChangeToStatus, TStatusNeighbIterator &statIter);
//		void ProcessStatusList(LayerType *InputList, LayerType *OutputList,
//	            StatusType ChangeToStatus, StatusType SearchForStatus, TStatusNeighbIterator &statusIt);
	ValueType UpdateActiveLayerValues(TimeStepType dt,
			MyLayerType *UpList, MyLayerType *DownList);
		            //, TValueNeighbIterator &outIt, TStatusNeighbIterator &statusIt);
		void PropagateLayerValues(StatusType from, StatusType to,
			                       StatusType promote, uint32 InOrOut);
			
	void ProcessOutsideList(MyLayerType *OutsideList, StatusType ChangeToStatus);//, TStatusNeighbIterator &statIter);
	void ProcessStatusList(MyLayerType *InputList, MyLayerType *OutputList,
            StatusType ChangeToStatus, StatusType SearchForStatus);//, TStatusNeighbIterator &statusIt);
private:
	
	
	
	ValueType CalculateUpdateValue(
		    const TimeStepType &dt,
		    const ValueType &value,
		    const ValueType &change)
		    {
			ValueType val = (value + dt * change); 
			return val;
			}
	
	
	void UnlinkNode(SparseFieldLevelSetNode *node, uint8 layerNum);
	void ReturnToNodeStore(SparseFieldLevelSetNode *node);
	void PushToLayer(SparseFieldLevelSetNode *node, uint8 layerNum);
	
	itk::SparseFieldCityBlockNeighborList< TRadius, TOffset, 3 > m_NeighborList;
	
	TValueNeighbIterator m_outIter;
	TStatusNeighbIterator m_statusIter;
	
	uint32 m_ElapsedIterations;
};

}
}
#endif /*APPLYUPDATECALCULATOR_H_*/
