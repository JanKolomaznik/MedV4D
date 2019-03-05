#ifndef ORGAN_SEGMENTATION_MODULE_HPP
#define ORGAN_SEGMENTATION_MODULE_HPP

#include <QtGui>
#include <QtCore>
#include "MedV4D/GUI/utils/Module.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "ViewerModule.hpp"
#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "MedV4D/Common/IDGenerator.h"

#include "MedV4D/Imaging/GraphCutSegmentation.h"

#include "DatasetManager.hpp"


class OrganSegmentationModule: public ViewerModule
{
public:
	typedef M4D::Imaging::Image< int16, 3 > Image16_3D;
	friend class OrganSegmentationWidget;
	OrganSegmentationModule()
		: ViewerModule( "Organ Segmentation Module" )
		, mVariance(1.0f)
		, mDatasetManager(nullptr)
	{}

	bool
	isUnloadable();

	void
	startSegmentationMode();

	void
	stopSegmentationMode();

	GraphCutSegmentationWrapper &
	getGraphCutSegmentationWrapper()
	{
		return mGraphCutSegmentationWrapper;
	}

	void
	update()
	{

	}

	void
	setDatasetManager(DatasetManager &aDatasetManager) override
	{
		mDatasetManager = &aDatasetManager;

	}

protected:
	M4D::Imaging::AImageDim<3>::ConstPtr
	getProcessedImage();

	void
	loadModule();

	void
	unloadModule();

	void
	createMask();

	void
	clearMask();

	void
	watershedTransformation();

	void
	fillMaskBorder(float aRadiusPercentage);

	void
	loadMask();

	void
	loadIndexFile();

	void
	computeStats();

	void
	loadModel();

	void
	updateTimestamp();

	void
	setVariance(float aVariance);

	void
	prepareMask( M4D::Imaging::Mask3D::Ptr aMask );

	void
	computeWatershedTransformation();

	void
	computeSegmentation();

	typedef M4D::Imaging::Image<int32_t, 3> WShedImage;

	OrganSegmentationController::Ptr mViewerController;
	M4D::Common::IDNumber mModeId;
	//TODO - only weak ptr used here
	M4D::Imaging::Mask3D::Ptr	mMask;
	M4D::Imaging::Mask3D::Ptr	mResult;
	Image16_3D::Ptr mImage;

	WShedImage::Ptr mWShedImage;

	float mVariance;

	GraphCutSegmentationWrapper	mGraphCutSegmentationWrapper;

	M4D::Imaging::CanonicalProbModel::Ptr mProbModel;

	M4D::GUI::IDMappingBuffer::Ptr mIDMappingBuffer;

	DatasetManager *mDatasetManager;
};


class StartOrganSegmentationAction: public QAction
{
	Q_OBJECT;
public:
	StartOrganSegmentationAction( OrganSegmentationModule &aModule, QObject *parent )
		: QAction( "Organ segmentation", parent ), mModule( aModule )
	{
		setCheckable( true );
		setChecked( false );
		QObject::connect( this, SIGNAL( toggled(bool) ), this, SLOT( toggleSegmentation(bool) ) );
	}
public slots:
	void
	toggleSegmentation( bool aToggle )
	{
		if( aToggle ) {
			mModule.startSegmentationMode();
		} else {
			mModule.stopSegmentationMode();
		}
	}

protected:
	OrganSegmentationModule &mModule;
};

#endif /*ORGAN_SEGMENTATION_MODULE_HPP*/
