#include "SegmentationManager.h"
#include "KidneySegmentationManager.h"
#include "ManualSegmentationManager.h"

SegmentationManagersList segmentationManagers;

void
InitializeSegmentationManagers()
{
	qRegisterMetaType<ManagerActivationInfo>();
	qRegisterMetaType<ResultsInfo>();

	ManualSegmentationManager::Instance().Initialize();
	KidneySegmentationManager::Instance().Initialize();

	segmentationManagers.push_back( static_cast<SegmentationManager*>(&(ManualSegmentationManager::Instance())) );
	segmentationManagers.push_back( static_cast<SegmentationManager*>(&(KidneySegmentationManager::Instance())) );
}

void
SegmentationManager::WantsProcessResults()
{
	emit ProcessResults( ResultsInfo( GetInputImage(), GetOutputGeometry() ) );
}

