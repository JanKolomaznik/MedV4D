#ifndef ANNOTATION_MODULE_H
#define ANNOTATION_MODULE_H

#include <QtGui>
#include <QtCore>
#include "MedV4D/GUI/utils/Module.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "AnnotationModule/AnnotationEditorController.hpp"
#include "MedV4D/Common/IDGenerator.h"

class AnnotationModule: public AModule
{
public:
	AnnotationModule(): AModule( "Annotation Module" )
	{}
	
	bool
	isUnloadable()
	{
		return false;
	}

protected:
	void
	loadModule()
	{
		ApplicationManager * appManager = ApplicationManager::getInstance();

		mViewerController = AnnotationEditorController::Ptr( new AnnotationEditorController );

		M4D::Common::IDNumber modeId = appManager->addNewMode( mViewerController/*controller*/, mViewerController/*renderer*/ );
		mViewerController->setModeId( modeId );
		QObject::connect( mViewerController.get(), SIGNAL( updateRequest() ), appManager, SLOT( updateGUIRequest() ) );
 
		appManager->createDockWidget( "Annotations", Qt::RightDockWidgetArea, mViewerController->getAnnotationView() );

		QList<QAction*> &annotationActions = mViewerController->getActions();
		QToolBar *toolbar = M4D::GUI::createToolbarFromActions( "Annotations toolbar", annotationActions );
		appManager->addToolBar( toolbar );

		mLoaded = true;
	}

	void
	unloadModule()
	{

	}
	
	AnnotationEditorController::Ptr mViewerController;
};

#endif /*ANNOTATION_MODULE_H*/
