#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtWidgets>
#include <memory>

#include "MedV4D/GUI/widgets/MultiDockWidget.h"
#include "MedV4D/GUI/widgets/AGLViewer.h"
#include "MedV4D/GUI/widgets/GeneralViewer.h"

namespace M4D
{
namespace GUI
{

class MainWindow: public QMainWindow
{
	Q_OBJECT;
public:
	MainWindow();

	MultiDockWidget *
	createDockWidget( QString aName, Qt::DockWidgetArea aArea, QWidget * aWidget, bool aDocked = true, bool aShow = true )
	{
		if (aWidget == nullptr) {
			_THROW_ M4D::ErrorHandling::ENULLPointer();
		}
		MultiDockWidget *dockwidget = new MultiDockWidget( aName, this );
		dockwidget->setWidget( aWidget );
		addDockWidget( aArea, dockwidget );
		//dockwidget->addDockingWindow( aArea, this );
		dockwidget->setVisible(aShow);
		dockwidget->setFloating(aDocked);
		return dockwidget;
	}

	virtual void
	addRenderingExtension( M4D::GUI::Viewer::RenderingExtension::Ptr aRenderingExtension ){}

	virtual void
	setViewerController( M4D::GUI::Viewer::AViewerController::Ptr aViewerController ){}
public slots:
	virtual void
	updateGui()
	{ update(); };
protected:

private:

};


} /*namespace GUI*/
} /*namespace M4D*/


#endif /*MAIN_WINDOW_H*/


