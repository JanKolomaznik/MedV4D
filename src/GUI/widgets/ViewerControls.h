#ifndef VIEWER_CONTROLS_H
#define VIEWER_CONTROLS_H

#include <QtGui>
#include <QtCore>
#include "tmp/ui_ViewerControls.h"
#include "GUI/utils/ApplicationManager.h"

class ViewerControls: public QWidget, public Ui::ViewerControls
{
	Q_OBJECT;
public:
	ViewerControls( QWidget *parent = NULL ): QWidget( parent )
	{
		setupUi( this );
		updateControls();
	}

public slots:
	void
	updateControls()
	{
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(pGenViewer != NULL) {
			setEnabled( true );
			Vector2f win = pGenViewer->getLUTWindow();
			windowCenterSpinBox->setValue( win[0] );
			windowWidthSpinBox->setValue( win[1] );
		} else {
			setEnabled( false );
		}
	}

	void
	windowChanged()
	{
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(pGenViewer != NULL) {
			pGenViewer->setLUTWindow( static_cast< float >( windowCenterSpinBox->value() ), static_cast< float >( windowWidthSpinBox->value() ) );
		}
	}

	void
	volumeRestrictionsChanged()
	{

	}

};

#endif /*VIEWER_CONTROLS_H*/
