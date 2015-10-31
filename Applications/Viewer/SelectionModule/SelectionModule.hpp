#pragma once

#include <QtGui>
#include <QtCore>
//#include "MedV4D/Common/Box.h"
#include "MedV4D/GUI/utils/Module.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"
#include "MedV4D/GUI/utils/DrawingMouseController.h"
#include "ViewerModule.hpp"

#include "DatasetManager.hpp"

class SelectionModule;

class BoxSelectionController
		: public ADrawingMouseController
{
	Q_OBJECT;
public:
	typedef std::shared_ptr<BoxSelectionController> Ptr;
	BoxSelectionController()
	{
	}


protected:
	void
	drawStep(const Vector3f &aStart, const Vector3f &aEnd, const Vector3f &aNormal) override
	{
		//drawMaskStep(aStart, aEnd, aNormal);
	}
};

class SelectionProxyController
		: public ModeProxyViewerController
		, public M4D::GUI::Viewer::RenderingExtension
{
	Q_OBJECT;
public:
	typedef std::shared_ptr<SelectionProxyController> Ptr;
	SelectionProxyController( SelectionModule &aModule )
		: mModule(aModule)
		, mOverlay(false)
		//, mBox(Vector3f(5, 5, 5), Vector3f(25, 25, 25))
	{
	}

	void
	activated() override
	{}

	void
	deactivated() override
	{}

	unsigned
	getAvailableViewTypes()const override;

	void
	render2DAlignedSlices( int32 aSliceIdx, Vector2f aInterval, CartesianPlanes aPlane ) override;

	void
	preRender3D() override;

	void
	postRender3D() override;

	void
	render3D();

	void
	setModeId( M4D::Common::IDNumber aId )
	{
		mModeId = aId;
	}
signals:
	void
	updateRequest();

protected:
	SelectionModule &mModule;
	bool mOverlay;
	M4D::Common::IDNumber mModeId;

	//AABox<3> mBox;
};

class SelectionModule: public ViewerModule
{
public:
	SelectionModule()
		: ViewerModule( "Selection Module" )
		, mDatasetManager(nullptr)
	{}

	bool
	isUnloadable() override
	{
		return false;
	}

	/*void
	update() override
	{

	}*/

	void
	setDatasetManager(DatasetManager &aDatasetManager) override
	{
		mDatasetManager = &aDatasetManager;
	}

protected:
	/*M4D::Imaging::AImageDim<3>::ConstPtr
	getProcessedImage();*/

	void
	loadModule() override;

	void
	unloadModule() override;

	void
	startBoxSelection();

	void
	endBoxSelection();

	SelectionProxyController::Ptr mViewerController;
	BoxSelectionController::Ptr mBoxSelectionController;

	M4D::Common::IDNumber mModeId;

	/*M4D::Imaging::Mask3D::Ptr	mMask;
	M4D::Imaging::Mask3D::Ptr	mResult;
	Image16_3D::Ptr mImage;

	GraphCutSegmentationWrapper	mGraphCutSegmentationWrapper;

	M4D::Imaging::CanonicalProbModel::Ptr mProbModel;

	M4D::GUI::IDMappingBuffer::Ptr mIDMappingBuffer;*/

	DatasetManager *mDatasetManager;
};
