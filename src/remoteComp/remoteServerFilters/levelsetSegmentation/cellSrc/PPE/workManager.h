#ifndef WORKMANAGER_H_
#define WORKMANAGER_H_

#include "itkObjectStore.h"
#include "itkSparseFieldLayer.h"
#include "../supportClasses.h"
#include "../SPE/configStructures.h"
#include "common/Thread.h"

namespace M4D
{
namespace Cell
{

template<typename IndexType, typename ValueType>
class WorkManager
{
public:
	typedef itk::SparseFieldLevelSetNode<IndexType> LayerNodeType;
	typedef itk::SparseFieldLayer<LayerNodeType> LayerType;

	struct LayerListType
	{
		typename LayerType::Pointer layers[LYERCOUNT];

		LayerListType()
		{
			for (uint32 i=0; i<LYERCOUNT; i++)
				layers[i] = LayerType::New();
		}
	};

	typedef ConfigStructures TConfigStructs;

	WorkManager(uint32 coreCount);
	~WorkManager();

	void PUSHNode(const IndexType &index, uint32 layerNum);
	void UNLINKNode(LayerNodeType *node, uint32 layerNum);

	void SetupRunConfig(RunConfiguration *conf);
	LayerListType *GetLayers()
	{
		return m_LayerSegments;
	}

	TConfigStructs *GetConfSructs()
	{
		return m_configs;
	}
	void InitCalculateChangeAndUpdActiveLayerConf();
	void InitPropagateValuesConf();
	void AllocateUpdateBuffers();
	
	void PrintLists(std::ostream &s);

private:
	typedef SparseFieldLevelSetNode NodeTypeInSPU;

	typedef itk::ObjectStore<LayerNodeType> LayerNodeStorageType;
	/** Container type used to store updates to the active layer. */
	typedef std::vector<ValueType> UpdateBufferType;

	/** Storage for layer node objects. */
	typename LayerNodeStorageType::Pointer m_LayerNodeStore;
	
	uint8 GetShortestLayer(uint8 layerNum);
	uint8 GetLongestLayer(uint8 layerNum);

	uint32 _numOfCores;

	// properties per SPE
	LayerListType *m_LayerSegments;
	UpdateBufferType *m_UpdateBuffers;
	TConfigStructs *m_configs;

	// access to layers
	M4D::Multithreading::Mutex _layerAccessMutex;

};

}
}

//include implementation
#include "src/workManager.tcc"

#endif /*WORKMANAGER_H_*/
