#include "BrushSettingsForm.hpp"
#include "ui_BrushSettingsForm.h"

#include <cassert>

BrushSettingsForm::BrushSettingsForm(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::BrushSettingsForm)
{
	ui->setupUi(this);

	QObject::connect(ui->mBrushTypeCombo, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &BrushSettingsForm::updated);
	QObject::connect(ui->mBrushRadiusSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, &BrushSettingsForm::updated);
}

BrushSettingsForm::~BrushSettingsForm()
{
	delete ui;
}

DrawingBrush BrushSettingsForm::brush() const
{
	assert(ui->mBrushTypeCombo->currentIndex() >= 0);
	return DrawingBrush {
		static_cast<BrushType>(ui->mBrushTypeCombo->currentIndex()),
		ui->mBrushRadiusSpinBox->value() };
}
