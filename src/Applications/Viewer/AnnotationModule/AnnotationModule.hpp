#ifndef ANNOTATION_MODULE_H
#define ANNOTATION_MODULE_H

#include <QtGui>
#include <QtCore>
#include "GUI/utils/Module.h"
#include "GUI/utils/ApplicationManager.h"
#include "AnnotationModule/AnnotationEditorController.hpp"
#include "common/IDGenerator.h"

class AnnotationModule: public AModule
{
public:

	void
	load()
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
	unload()
	{

	}

	bool
	isUnloadable()
	{
		return false;
	}

	std::string
	getName()
	{
		return "Annotation Module";
	}
protected:
	AnnotationEditorController::Ptr mViewerController;
};

#endif /*ANNOTATION_MODULE_H*/
