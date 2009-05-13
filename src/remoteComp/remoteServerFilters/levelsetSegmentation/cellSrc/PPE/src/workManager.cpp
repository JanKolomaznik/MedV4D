//#ifndef WORKMANAGER_H_
//#error File workManager.tcc cannot be included directly!
//#else
//
//namespace M4D {
//namespace Cell {

#include "common/Common.h"
#include "../workManager.h"

#ifdef FOR_CELL
#include <libspe2.h>
#include <libmisc.h> 
#endif

using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////

WorkManager::WorkManager(uint32 coreCount, RunConfiguration *rc)
	: _numOfCores(coreCount), _runConf(rc)
{
	// aloc props related to SPEs
	m_LayerSegments = new LayerListType[_numOfCores];
	m_UpdateBuffers = new UpdateBufferType[_numOfCores];
	
#ifdef FOR_CELL
	_configs = (ConfigStructures *) malloc_align(
			_numOfCores, 7);
	_calcChngApplyUpdateConf = (CalculateChangeAndUpdActiveLayerConf *)
		malloc_align(_numOfCores, CalculateChangeAndUpdActiveLayerConf_AllignExponent);
	_propagateValsConf = (PropagateValuesConf *) malloc_align(
			_numOfCores, PropagateValuesConf_AllignExponent);
#else
	_configs = new ConfigStructures[_numOfCores];
	_calcChngApplyUpdateConf = new CalculateChangeAndUpdActiveLayerConf[_numOfCores];
	_propagateValsConf = new PropagateValuesConf[_numOfCores];
#endif
	
	for(uint32 spuIt=0; spuIt<_numOfCores; spuIt++)
	{
		_configs[spuIt].runConf = _runConf;
		_configs[spuIt].calcChngApplyUpdateConf = &_calcChngApplyUpdateConf[spuIt];
		_configs[spuIt].propagateValsConf = &_propagateValsConf[spuIt];
	}
	
	// layer storeage init
	m_LayerNodeStore = LayerNodeStorageType::New();
	  m_LayerNodeStore->SetGrowthStrategyToExponential();
}

///////////////////////////////////////////////////////////////////////////////

WorkManager::~WorkManager()
{
#ifdef FOR_CELL
	free_align(_configs);
	free_align(_calcChngApplyUpdateConf);
	free_align(_propagateValsConf);
#else
	delete [] _configs;
	delete [] _calcChngApplyUpdateConf;
	delete [] _propagateValsConf;
#endif
	delete [] m_UpdateBuffers;
	delete [] m_LayerSegments;
}

///////////////////////////////////////////////////////////////////////////////


void
WorkManager
	::PUSHNode(const TIndex &index, uint32 layerNum)
{
	M4D::Multithreading::ScopedLock lock(_layerAccessMutex);
	
	LayerNodeType *node = m_LayerNodeStore->Borrow();
	node->m_Value = index;
	// push node into segment that have the least count of nodes
	m_LayerSegments[GetShortestLayer(layerNum)].layers[layerNum].PushFront( node );
}

///////////////////////////////////////////////////////////////////////////////

void
WorkManager::UNLINKNode(LayerNodeType *node, uint32 layerNum, uint32 segmentID)
{
	//M4D::Multithreading::ScopedLock lock(_layerAccessMutex);
	
	// unlink node from segment that have the biggest count of nodes
	m_LayerSegments[segmentID].layers[layerNum].Unlink(node);
}

///////////////////////////////////////////////////////////////////////////////

void
WorkManager::InitCalculateChangeAndUpdActiveLayerConf()
{
	for(uint32 spuIt=0; spuIt<_numOfCores; spuIt++)
	{
		_calcChngApplyUpdateConf[spuIt].layer0Begin = 
			m_LayerSegments[spuIt].layers[0].Front();
		_calcChngApplyUpdateConf[spuIt].layer0End = 
			m_LayerSegments[spuIt].layers[0].End();
    
		_calcChngApplyUpdateConf[spuIt].updateBuffBegin = 
			&m_UpdateBuffers[spuIt][0];
	}
}

///////////////////////////////////////////////////////////////////////////////

void
WorkManager::InitPropagateValuesConf()
{
	for(uint32 spuIt=0; spuIt<_numOfCores; spuIt++)
	{
		for(uint32 i=0; i<LYERCOUNT; i++)
	    {
			_propagateValsConf[spuIt].layerBegins[i] = 
				m_LayerSegments[spuIt].layers[i].Front();
			
			_propagateValsConf[spuIt].layerEnds[i] = 
				m_LayerSegments[spuIt].layers[i].End();
	    }
	}
}

///////////////////////////////////////////////////////////////////////////////
//
//void
//WorkManager::SetupRunConfig(RunConfiguration *conf)
//{
//	// copy it into configs of all SPEs 
//	for(uint32 spuIt=0; spuIt<_numOfCores; spuIt++)
//	{
//		m_configs[spuIt].runConf = *conf;
//	}
//}

///////////////////////////////////////////////////////////////////////////////


void
WorkManager::AllocateUpdateBuffers()
{
  // Preallocate the update buffer.  NOTE: There is currently no way to
  // downsize a std::vector. This means that the update buffer will grow
  // dynamically but not shrink.  In newer implementations there may be a
  // squeeze method which can do this.  Alternately, we can implement our own
  // strategy for downsizing.
//	if(m_Layers[0]->Size() > 10000)
//	{
//		int i = 10;
//	}
	for(uint32 spuIt=0; spuIt<_numOfCores; spuIt++)
	{	
		m_UpdateBuffers[spuIt].clear();
		m_UpdateBuffers[spuIt].reserve(m_LayerSegments[spuIt].layers[0].Size());
		memset(&m_UpdateBuffers[spuIt][0], 0, m_LayerSegments[spuIt].layers[0].Size() * sizeof(ValueType));
	}
//  std::cout << "Update list, after reservation:" << std::endl;
//  PrintUpdateBuf(std::cout);
}

///////////////////////////////////////////////////////////////////////////////


void
WorkManager::PrintLists(std::ostream &s, bool withMembers)
{
	LayerNodeType *begin;
	const LayerNodeType *end;
	
	for(uint32 coreIt=0; coreIt<_numOfCores; coreIt++)
	{
		s << "Core" << coreIt 
			<< " :::::::::::::::::::::::::::::::::::::" << std::endl;
		for(uint32 i=0; i<LYERCOUNT; i++)
		{
			s << "layer" << i << ", size=" << 
				m_LayerSegments[coreIt].layers[i].Size() << std::endl;
			
			if(withMembers)
			{
				begin = m_LayerSegments[coreIt].layers[i].Front();
				end = m_LayerSegments[coreIt].layers[i].End();
				
				while(begin != end)
				{
					s << begin->m_Value << std::endl;
					begin = (LayerNodeType *)begin->Next.Get64();
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////


uint8
WorkManager::GetShortestLayer(uint8 layerNum)
{
	uint8 shortest = 0;
	for(uint32 i=1; i<_numOfCores; i++)
	{
		if(m_LayerSegments[i].layers[layerNum].Size() < 
				m_LayerSegments[shortest].layers[layerNum].Size())
			shortest = i;
	}
	return shortest;
}

///////////////////////////////////////////////////////////////////////////////


uint8
WorkManager::GetLongestLayer(uint8 layerNum)
{
	uint8 longest = 0;
	for(uint32 i=1; i<_numOfCores; i++)
	{
		if(m_LayerSegments[i].layers[layerNum].Size() > 
			m_LayerSegments[longest].layers[layerNum].Size())
			longest = i;
	}
	return longest;
}

///////////////////////////////////////////////////////////////////////////////

//template<class TInputImage,class TFeatureImage, class TOutputPixelType>
//void
//MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
//::PrintUpdateBuf(std::ostream &s)
//{
//	s << "size=" << m_Layers[0]->Size() << std::endl;
//	ValueType *updBuffData = &m_UpdateBuffer[0];
//    for(uint32 i=0; i<m_Layers[0]->Size(); i++, updBuffData++)
//    	s << *updBuffData << ", ";
//  	  s << std::endl;
//}

///////////////////////////////////////////////////////////////////////////////

//}
//}
//#endif
