#pragma once

#include <QWidget>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QHBoxLayout>
#include <cmath>

#include "MedV4D/GUI/utils/QtM4DTools.h"

class DoubleSpinBoxWithSlider: public QWidget
{
	Q_OBJECT;
public:
	DoubleSpinBoxWithSlider(QWidget *parent = nullptr)
		: QWidget( parent )
	{
		auto layout = new QHBoxLayout;
		layout->addWidget(mSlider = new QSlider);
		layout->addWidget(mSpinBox = new QDoubleSpinBox);
		layout->setStretch(0, 2);
		layout->setMargin(0);
		setLayout(layout);

		mSlider->setOrientation(Qt::Horizontal);
		setMinimum(0.0);
		setMaximum(5000.0);
		setDecimals(2);

		setValue(1000.0);

		QObject::connect(mSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &DoubleSpinBoxWithSlider::spinBoxUpdated);
		QObject::connect(mSlider, &QSlider::valueChanged, this, &DoubleSpinBoxWithSlider::sliderUpdated);
	}
	double
	value() const
	{
		return mValue;
	}
public slots:
	void
	setMinimum(double aMinimum)
	{
		mMinimum = aMinimum;
		updateWidgets();
	}

	void
	setMaximum(double aMaximum)
	{
		mMaximum = aMaximum;
		updateWidgets();
	}

	void
	setDecimals(int aDecimals)
	{
		mDecimals = aDecimals;
		updateWidgets();
	}

	void
	setValue(double aValue)
	{
		mValue = aValue;
		updateWidgets();
		emit valueChanged(mValue);
	}
signals:
	void
	valueChanged(double aValue);

protected slots:
	void
	sliderUpdated()
	{
		auto val = mSlider->value();
		auto multiplier = std::pow(10.0, mDecimals);
		mValue = val / multiplier;
		updateWidgets();
		emit valueChanged(mValue);
	}

	void
	spinBoxUpdated()
	{
		mValue = mSpinBox->value();
		updateWidgets();
		emit valueChanged(mValue);
	}

protected:
	void
	updateWidgets()
	{
		M4D::GUI::QtSignalBlocker blocker1(mSpinBox);
		M4D::GUI::QtSignalBlocker blocker2(mSlider);

		mSpinBox->setMinimum(mMinimum);
		mSpinBox->setMaximum(mMaximum);
		mSpinBox->setDecimals(mDecimals);

		mSpinBox->setValue(mValue);

		auto multiplier = std::pow(10.0, mDecimals);
		mSlider->setMinimum(std::floor(mMinimum * multiplier));
		mSlider->setMaximum(std::ceil(mMaximum * multiplier));
		mSlider->setValue(std::round(mValue * multiplier));
	}

	double mMinimum;
	double mMaximum;
	int mDecimals;
	double mValue;
	QSlider *mSlider;
	QDoubleSpinBox *mSpinBox;

};


