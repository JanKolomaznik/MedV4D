#ifndef VIEWER_CONTROLS_H
#define VIEWER_CONTROLS_H

#include <QtGui>
#include <QtCore>
#include "MedV4D/generated/ui_ViewerControls.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"

class ViewerControls: public QWidget, public Ui::ViewerControls
{
	Q_OBJECT;
public:
	ViewerControls( QWidget *parent = NULL ): QWidget( parent ), mUpdating( false )
	{
		setupUi( this );
		updateControls();
	}

public slots:

	void
	resetVolumeRestrictions()
	{
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(pGenViewer != NULL) {
			pGenViewer->setVolumeRestrictions( 
					Vector2f( 0.0f, 1.0f ), 
					Vector2f( 0.0f, 1.0f ), 
					Vector2f( 0.0f, 1.0f ) 
					);
		}
	}

	void
	updateControls()
	{
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(pGenViewer != NULL) {
			setEnabled( true );
			
			mUpdating = true;
			
			Vector2f win = pGenViewer->getLUTWindow();
			windowCenterSpinBox->setValue( win[0] );
			windowWidthSpinBox->setValue( win[1] );

			Vector2f x, y, z;
			pGenViewer->getVolumeRestrictions(x,y,z);
			xIntervalASpinBox->setValue(x[0]); xIntervalBSpinBox->setValue(x[1]);
			yIntervalASpinBox->setValue(y[0]); yIntervalBSpinBox->setValue(y[1]);
			zIntervalASpinBox->setValue(z[0]); zIntervalBSpinBox->setValue(z[1]);

			volumeRestrictionsGroupBox->setChecked( pGenViewer->isVolumeRestrictionEnabled() );
			
			Vector2u grid = pGenViewer->getTiling();
			viewportTilesRows->setValue( grid[0] );
			viewportTilesCols->setValue( grid[1] );
			sliceStep->setValue( pGenViewer->getTilingSliceStep() );
		} else {
			setEnabled( false );
		}
		mUpdating = false;
	}

	void
	windowChanged()
	{
		if (mUpdating) return;
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
		if (mUpdating) return;
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(pGenViewer != NULL) {
			pGenViewer->setVolumeRestrictions( 
					volumeRestrictionsGroupBox->isChecked(),
					Vector2f( static_cast<float>(xIntervalASpinBox->value()), static_cast<float>(xIntervalBSpinBox->value()) ), 
					Vector2f( static_cast<float>(yIntervalASpinBox->value()), static_cast<float>(yIntervalBSpinBox->value()) ), 
					Vector2f( static_cast<float>(zIntervalASpinBox->value()), static_cast<float>(zIntervalBSpinBox->value()) ) 
					);
		}
	}
	
	void
	viewportTilingChanged()
	{
		if (mUpdating) return;
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *pGenViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(pGenViewer != NULL) {
			pGenViewer->setTiling( viewportTilesRows->value(), viewportTilesCols->value(), sliceStep->value() );
		}
	}
private:
	bool mUpdating;
};

#endif /*VIEWER_CONTROLS_H*/
