#include <QtGui>

#include "transitionFunctionRenderArea.h"
#include "settingsBoxWidget.h"

const int IdRole = Qt::UserRole;

SettingsBoxWidget::SettingsBoxWidget(transitionFunction* functionData, QWidget* parent) : QWidget(parent)
{
	setBackgroundRole(QPalette::Base);
	setAutoFillBackground(true);
	
	this->functionData = functionData;
	
	renderArea = new transitionFunctionRenderAreaWidget(functionData);
	
	pointSpinBox = new QSpinBox();
	pointSpinBox->setRange(functionData->GetMinPoint(), functionData->GetMaxPoint());
	
	pointLabel = new QLabel(tr("&Point number:"));
	pointLabel->setBuddy(pointSpinBox);

	valueSpinBox = new QDoubleSpinBox();
	valueSpinBox->setRange(functionData->GetValueOfMinPoint(), functionData->GetValueOfMaxPoint());
	valueSpinBox->setDecimals(3);
	valueSpinBox->setSingleStep(0.01);

	valueLabel = new QLabel(tr("&Value of point:"));
	valueLabel->setBuddy(valueSpinBox);
    
	addPointButton = new QPushButton(tr("&Add point"));

	resetTransitionFunctionButton = new QPushButton(tr("&Reset"));

    connect(resetTransitionFunctionButton, SIGNAL(clicked()), this, SLOT(resetDemandedSlot()));
	connect(addPointButton, SIGNAL(clicked()), this, SLOT(pointAddedSlot()));
	
    QGridLayout *mainLayout = new QGridLayout;

    mainLayout->setColumnStretch(0, 1);
    mainLayout->setColumnStretch(3, 1);
    mainLayout->addWidget(renderArea, 0, 0, 1, 4);
    mainLayout->setRowMinimumHeight(1, 6);
    mainLayout->addWidget(pointLabel, 2, 1, Qt::AlignRight);
    mainLayout->addWidget(pointSpinBox, 2, 2);
    mainLayout->addWidget(valueLabel, 3, 1, Qt::AlignRight);
    mainLayout->addWidget(valueSpinBox, 3, 2);
    mainLayout->addWidget(addPointButton, 4, 1, Qt::AlignCenter);
    mainLayout->addWidget(resetTransitionFunctionButton, 5, 1, Qt::AlignCenter);
    setLayout(mainLayout);

	functionChangedSlot();

    setWindowTitle(tr("HapticInterface SettingsBox"));
}

void SettingsBoxWidget::resetDemandedSlot()
{
	emit resetFunction();
}

void SettingsBoxWidget::pointAddedSlot()
{
	functionData->SetValueOnPoint((unsigned short)pointSpinBox->value(), valueSpinBox->value());
	functionChangedSlot();
}

void SettingsBoxWidget::functionChangedSlot()
{
	pointSpinBox->setRange(functionData->GetMinPoint(), functionData->GetMaxPoint());
	renderArea->update();
}
