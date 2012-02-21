#ifndef ORGAN_SEGMENTATION_MODULE_HPP
#define ORGAN_SEGMENTATION_MODULE_HPP

#include <QtGui>
#include <QtCore>
#include "MedV4D/GUI/utils/Module.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "OrganSegmentationModule/OrganSegmentationWidget.hpp"
#include "OrganSegmentationModule/OrganSegmentationController.hpp"
#include "MedV4D/Common/IDGenerator.h"




class OrganSegmentationModule: public AModule
{
public:
	friend class OrganSegmentationWidget;
	OrganSegmentationModule(): AModule( "Organ Segmentation Module" )
	{}

	bool
	isUnloadable();

	void
	startSegmentationMode();

	void
	stopSegmentationMode();
protected:
	void
	loadModule();

	void
	unloadModule();
	
	void 
	createMask();
	
	void
	updateTimestamp();
	
	OrganSegmentationController::Ptr mViewerController;
	M4D::Common::IDNumber mModeId;
	
	M4D::Imaging::Mask3D::Ptr	mMask;
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
