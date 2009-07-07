#ifndef WORKMANAGER_H_
#define WORKMANAGER_H_

#include "objectStore.h"
#include "../SPE/tools/sparesFieldLayer.h"
#include "../supportClasses.h"
#include "../SPE/configStructures.h"
#include "common/Thread.h"
#include "updateValsAllocator.h"

namespace M4D
{
namespace Cell
{

class WorkManager
{
public:
	typedef SparseFieldLevelSetNode LayerNodeType;
	typedef SparseFieldLayer<LayerNodeType> LayerType;

	struct LayerListType
	{
		LayerType layers[LYERCOUNT];
	};

	WorkManager(uint32 coreCount, RunConfiguration *rc);
	~WorkManager();

	void PUSHNode(const TIndex &index, uint32 layerNum);
	void UNLINKNode(LayerNodeType *node, uint32 layerNum, uint32 segmentID);

	LayerListType *GetLayers()
	{
		return m_LayerSegments;
	}

	ConfigStructures *GetConfSructs()
	{
		return _configs;
	}
	void InitCalculateChangeAndUpdActiveLayerConf();
	void InitPropagateValuesConf();
	void AllocateUpdateBuffers();
	
	void CheckLayerSizes();
	
	void PrintLists(std::ostream &s, bool withMembers);
	
	uint32 GetLayer0TotalSize(){
		uint32 size = 0;
		for(uint32 i=0; i<_numOfCores; i++)
			size += m_LayerSegments[i].layers[0].Size();
		
		return size;
	}
	
	TimeStepType _dt;

	typedef float32 ValueType;
#define NodeStoreChunkSize 2048
	typedef PPEObjectStore<LayerNodeType, NodeStoreChunkSize> LayerNodeStorageType;
	/** Container type used to store updates to the active layer. */
	typedef UpdateValsAllocator<ValueType> UpdateBufferType;	

	/** Storage for layer node objects. */
	LayerNodeStorageType m_LayerNodeStore;
	
	uint8 GetShortestLayer(uint8 layerNum);
	uint8 GetLongestLayer(uint8 layerNum);

	uint32 _numOfCores;

	// properties per SPE
	LayerListType *m_LayerSegments;
	UpdateBufferType *m_UpdateBuffers;
	
	ConfigStructures *_configs;
	const RunConfiguration *_runConf;
	CalculateChangeAndUpdActiveLayerConf *_calcChngApplyUpdateConf;
	PropagateValuesConf *_propagateValsConf;

	// access to layers
	M4D::Multithreading::Mutex _layerAccessMutex;

};

}
}

#endif /*WORKMANAGER_H_*/
