#ifndef VIEWER_CONTROLS_H
#define VIEWER_CONTROLS_H

#include <QtWidgets>
#include <QtCore>
#include "MedV4D/generated/ui_ViewerControls.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "MedV4D/GUI/utils/QtM4DTools.h"

class ViewerControls: public QWidget, public Ui::ViewerControls
{
	Q_OBJECT;
public:
	ViewerControls(QWidget *parent = nullptr)
		: QWidget( parent )
		, mUpdating( false )
		, mCurrentViewer(nullptr)
	{
		setupUi( this );
		updateControls();
	}

	void
	setViewer(M4D::GUI::Viewer::GeneralViewer *aViewer)
	{
		mCurrentViewer = aViewer;
		updateControls();
	}

public slots:

	void
	resetVolumeRestrictions()
	{
		if(mCurrentViewer) {
			mCurrentViewer->setVolumeRestrictions(
					Vector2f( 0.0f, 1.0f ),
					Vector2f( 0.0f, 1.0f ),
					Vector2f( 0.0f, 1.0f )
					);
		}
	}

	void
	updateControls()
	{
		if (mUpdating) {
			return;
		}
		if (!mCurrentViewer) {
			setEnabled(false);
			return;
		}

		setEnabled(true);
		M4D::GUI::QtSignalBlocker signalBlocker(this);
		mUpdating = true;

		if (mCurrentViewer->getViewType() == M4D::GUI::Viewer::vt3D) {
			mViewTypeTabWidget->setCurrentIndex(1);
		} else {
			mViewTypeTabWidget->setCurrentIndex(0);
		}

		glm::fvec2 win = mCurrentViewer->getLUTWindow();
		windowCenterSpinBox->setValue( win[0] );
		windowWidthSpinBox->setValue( win[1] );

		Vector2f x, y, z;
		mCurrentViewer->getVolumeRestrictions(x,y,z);
		xIntervalASpinBox->setValue(x[0]); xIntervalBSpinBox->setValue(x[1]);
		yIntervalASpinBox->setValue(y[0]); yIntervalBSpinBox->setValue(y[1]);
		zIntervalASpinBox->setValue(z[0]); zIntervalBSpinBox->setValue(z[1]);

		volumeRestrictionsGroupBox->setChecked( mCurrentViewer->isVolumeRestrictionEnabled() );

		Vector2u grid = mCurrentViewer->getTiling();
		viewportTilesRows->setValue( grid[0] );
		viewportTilesCols->setValue( grid[1] );
		sliceStep->setValue(int(mCurrentViewer->getTilingSliceStep()));

		mUpdating = false;
	}

protected slots:
	void
	settingsChanged()
	{
		if (mUpdating || !mCurrentViewer) {
			return;
		}

		mUpdating = true;

		if (mViewTypeTabWidget->currentIndex() == 1) {
			mCurrentViewer->setViewType(M4D::GUI::Viewer::vt3D);
		} else {
			mCurrentViewer->setViewType(M4D::GUI::Viewer::vt2DAlignedSlices);
		}

		mCurrentViewer->setLUTWindow( static_cast< float >( windowCenterSpinBox->value() ), static_cast< float >( windowWidthSpinBox->value() ) );

		mCurrentViewer->setVolumeRestrictions(
				volumeRestrictionsGroupBox->isChecked(),
				Vector2f( static_cast<float>(xIntervalASpinBox->value()), static_cast<float>(xIntervalBSpinBox->value()) ),
				Vector2f( static_cast<float>(yIntervalASpinBox->value()), static_cast<float>(yIntervalBSpinBox->value()) ),
				Vector2f( static_cast<float>(zIntervalASpinBox->value()), static_cast<float>(zIntervalBSpinBox->value()) )
				);

		mCurrentViewer->setTiling( viewportTilesRows->value(), viewportTilesCols->value(), sliceStep->value() );

		mUpdating = false;
		updateControls();
	}

	/*void
	windowChanged()
	{
		if (mUpdating) return;
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *mCurrentViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(mCurrentViewer != nullptr) {
			mCurrentViewer->setLUTWindow( static_cast< float >( windowCenterSpinBox->value() ), static_cast< float >( windowWidthSpinBox->value() ) );
		}
	}*/

	/*void
	volumeRestrictionsChanged()
	{
		if (mUpdating) return;
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *mCurrentViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(mCurrentViewer != nullptr) {
			mCurrentViewer->setVolumeRestrictions(
					volumeRestrictionsGroupBox->isChecked(),
					Vector2f( static_cast<float>(xIntervalASpinBox->value()), static_cast<float>(xIntervalBSpinBox->value()) ),
					Vector2f( static_cast<float>(yIntervalASpinBox->value()), static_cast<float>(yIntervalBSpinBox->value()) ),
					Vector2f( static_cast<float>(zIntervalASpinBox->value()), static_cast<float>(zIntervalBSpinBox->value()) )
					);
		}
	}*/

	/*void
	viewportTilingChanged()
	{
		if (mUpdating) return;
		M4D::GUI::Viewer::AGLViewer *pViewer;
		pViewer = ViewerManager::getInstance()->getSelectedViewer();

		M4D::GUI::Viewer::GeneralViewer *mCurrentViewer = dynamic_cast<M4D::GUI::Viewer::GeneralViewer*> (pViewer);
		if(mCurrentViewer != nullptr) {
			mCurrentViewer->setTiling( viewportTilesRows->value(), viewportTilesCols->value(), sliceStep->value() );
		}
	}*/
protected:
	M4D::GUI::Viewer::GeneralViewer *mCurrentViewer;
private:
	bool mUpdating;
};

#endif /*VIEWER_CONTROLS_H*/
