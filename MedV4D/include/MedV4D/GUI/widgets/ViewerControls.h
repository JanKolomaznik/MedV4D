#ifndef VIEWER_CONTROLS_H
#define VIEWER_CONTROLS_H

#include <QtWidgets>
#include <QtCore>
#include "MedV4D/generated/ui_ViewerControls.h"
#include "MedV4D/GUI/managers/ApplicationManager.h"
#include "MedV4D/GUI/utils/QtM4DTools.h"

#include <QPushButton>

/*class ColorButton : public QPushButton
{
	Q_OBJECT;
public:
	ColorButton(QWidget *parent = nullptr)
		: QPushButton(parent)
	{
	}

	QColor
	color() const
	{
		return mColor;
	}

	QColor getIdealTextColor(const QColor& aBackgroundColor) const
	{
		const int cThreshold = 105;
		int backgroundDelta = (aBackgroundColor.red() * 0.299) + (aBackgroundColor.green() * 0.587) + (aBackgroundColor.blue() * 0.114);
		return QColor((255- backgroundDelta < cThreshold) ? Qt::black : Qt::white);
	}

public slots:
	void
	setColor(QColor aColor)
	{
		mColor = aColor;

		static const QString cColorStyle("QPushButton { background-color : %1; color : %2; }");
		QColor idealTextColor = getIdealTextColor(aColor);
		setStyleSheet(cColorStyle.arg(aColor.name()).arg(idealTextColor.name()));
	}
protected:
	QColor mColor;
};*/

class ViewerControls: public QWidget, public Ui::ViewerControls
{
	Q_OBJECT;
public:
	ViewerControls(QWidget *parent = nullptr)
		: QWidget( parent )
		, mCurrentViewer(nullptr)
		, mUpdating( false )
	{
		setupUi( this );
		QObject::connect(mIsoSurfacesSetup, &IsoSurfacesSetupWidget::isoSurfaceSetupUpdated, this, &ViewerControls::settingsChanged);
		updateControls();
	}

	void
	setViewer(M4D::GUI::Viewer::GeneralViewer *aViewer)
	{
		mCurrentViewer = aViewer;
		updateControls();
	}

	M4D::GUI::Viewer::GeneralViewer *
	viewer() const
	{
		return mCurrentViewer;
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

    if (mViewTypeTabWidget->currentIndex() != 2) // eigenvalues tab - this one is not concerned about 2D/3D
    {
      if (mCurrentViewer->getViewType() == M4D::GUI::Viewer::vt3D) {
        mViewTypeTabWidget->setCurrentIndex(1);
      }
      else {
        mViewTypeTabWidget->setCurrentIndex(0);
      }
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

		mEnablePreintegratedTFCheckBox->setChecked(mCurrentViewer->isIntegratedTransferFunctionEnabled());

		//mIsoValueSlider;
		std::cout << mCurrentViewer->isoSurfaces()[0].isoSurfaceColor[0] << std::endl;
		mIsoSurfacesSetup->setIsoSurfaces(mCurrentViewer->isoSurfaces());
		//mIsoValueSetter->setValue(mCurrentViewer->isoSurfaceValue());
		//mSurfaceColorButton->setColor(mCurrentViewer->isoSurfaceColor());

    vorgl::EigenvaluesRenderingOptions eigenvaluesOptions = this->mCurrentViewer->getEigenvaluesOptions();
    this->alphaSpinBox->setValue(eigenvaluesOptions.eigenvaluesConstants[0]);
    this->betaSpinBox->setValue(eigenvaluesOptions.eigenvaluesConstants[1]);
    this->gammaSpinBox->setValue(eigenvaluesOptions.eigenvaluesConstants[2]);
    this->objectnessComboBox->setCurrentIndex(eigenvaluesOptions.objectnessType);

    this->franghisVesselnessRadioButton->setChecked(eigenvaluesOptions.eigenvaluesType == 0);
    this->satosVesselessRadioButton->setChecked(eigenvaluesOptions.eigenvaluesType == 1);
    this->kollersVesselnessRadioButton->setChecked(eigenvaluesOptions.eigenvaluesType == 2);
    this->objectnessRadioButton->setChecked(eigenvaluesOptions.eigenvaluesType == 3);
    this->customVesselnessRadioButton->setChecked(eigenvaluesOptions.eigenvaluesType == 4);

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

    if (mViewTypeTabWidget->currentIndex() != 2) // eigenvalues tab - this one is not concerned about 2D/3D
    {
      if (mViewTypeTabWidget->currentIndex() == 1) {
        if (mCurrentViewer->getViewType() != M4D::GUI::Viewer::vt3D) {
          mCurrentViewer->setViewType(M4D::GUI::Viewer::vt3D);
        }
      }
      else {
        if (mCurrentViewer->getViewType() != M4D::GUI::Viewer::vt2DAlignedSlices) {
          mCurrentViewer->setViewType(M4D::GUI::Viewer::vt2DAlignedSlices);
        }
      }
    }

		mCurrentViewer->setLUTWindow( static_cast< float >( windowCenterSpinBox->value() ), static_cast< float >( windowWidthSpinBox->value() ) );

		mCurrentViewer->setVolumeRestrictions(
				volumeRestrictionsGroupBox->isChecked(),
				Vector2f( static_cast<float>(xIntervalASpinBox->value()), static_cast<float>(xIntervalBSpinBox->value()) ),
				Vector2f( static_cast<float>(yIntervalASpinBox->value()), static_cast<float>(yIntervalBSpinBox->value()) ),
				Vector2f( static_cast<float>(zIntervalASpinBox->value()), static_cast<float>(zIntervalBSpinBox->value()) )
				);

		mCurrentViewer->setTiling( viewportTilesRows->value(), viewportTilesCols->value(), sliceStep->value() );

		mCurrentViewer->enableIntegratedTransferFunction(mEnablePreintegratedTFCheckBox->isChecked());

    int type = 0;
    if (this->franghisVesselnessRadioButton->isChecked())
    {
      type = 0;
    }
    else if (this->satosVesselessRadioButton->isChecked())
    {
      type = 1;
    }
    else if (this->kollersVesselnessRadioButton->isChecked())
    {
      type = 2;
    }
    else if (this->objectnessRadioButton->isChecked())
    {
      type = 3;
    }
    else if (this->customVesselnessRadioButton->isChecked())
    {
      type = 4;
    }

    mCurrentViewer->setEigenvaluesOptions(type, this->alphaSpinBox->value(), this->betaSpinBox->value(), this->gammaSpinBox->value(), this->objectnessComboBox->currentIndex());

    mCurrentViewer->setLUTWindow(this->windowCenterSpinBox->value(), this->windowWidthSpinBox->value());

		mCurrentViewer->setIsoSurfaces(mIsoSurfacesSetup->isoSurfaces());
		//mCurrentViewer->setIsoSurfaceValue(mIsoValueSetter->value());
		//mCurrentViewer->setIsoSurfaceColor(mSurfaceColorButton->color());

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
