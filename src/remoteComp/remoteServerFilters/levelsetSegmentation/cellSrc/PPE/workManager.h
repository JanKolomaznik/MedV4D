#ifndef WORKMANAGER_H_
#define WORKMANAGER_H_

#include "itkObjectStore.h"
//#include "itkSparseFieldLayer.h"
#include "../SPE/tools/sparesFieldLayer.h"
#include "../supportClasses.h"
#include "../SPE/configStructures.h"
#include "common/Thread.h"

namespace M4D
{
namespace Cell
{

//template<typename IndexType, typename ValueType>
class WorkManager
{
public:
	//typedef itk::SparseFieldLevelSetNode<IndexType> LayerNodeType;
	typedef SparseFieldLevelSetNode LayerNodeType;
	typedef M4D::Cell::SparseFieldLayer<LayerNodeType> LayerType;

	struct LayerListType
	{
		LayerType layers[LYERCOUNT];
	};

	WorkManager(uint32 coreCount, RunConfiguration *rc);
	~WorkManager();

	void PUSHNode(const TIndex &index, uint32 layerNum);
	void UNLINKNode(LayerNodeType *node, uint32 layerNum, uint32 segmentID);

	//void SetupRunConfig(RunConfiguration *conf);
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
	
	void PrintLists(std::ostream &s, bool withMembers);
	
	uint32 GetLayer0TotalSize(){
		uint32 size = 0;
		for(uint32 i=0; i<_numOfCores; i++)
			size += m_LayerSegments[i].layers[0].Size();
		
		return size;
	}
	
	TimeStepType _dt;

private:
	typedef float32 ValueType;
	//typedef SparseFieldLevelSetNode NodeTypeInSPU;

	typedef itk::ObjectStore<LayerNodeType> LayerNodeStorageType;
	/** Container type used to store updates to the active layer. */
	typedef std::vector<ValueType> UpdateBufferType;
	

	/** Storage for layer node objects. */
	LayerNodeStorageType::Pointer m_LayerNodeStore;
	
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

////include implementation
//#include "src/workManager.tcc"

#endif /*WORKMANAGER_H_*/
