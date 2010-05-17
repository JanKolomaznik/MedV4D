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
	connect(renderArea, SIGNAL(addPointSignal(double, double)), this, SLOT(pointAddedSlot(double, double)));

	resetTransitionFunctionButton = new QPushButton(tr("&Reset function"));

	zoomInButton = new QPushButton(tr("Zoom &in"));
	zoomOutButton = new QPushButton(tr("Zoom &out"));
	
	hapticLabel = new QLabel(tr("Haptic function:"));

    connect(resetTransitionFunctionButton, SIGNAL(clicked()), this, SLOT(resetDemandedSlot()));
	connect(zoomInButton, SIGNAL(clicked()), this, SLOT(zoomInHapticSlot()));
	connect(zoomOutButton, SIGNAL(clicked()), this, SLOT(zoomOutHapticSlot()));
	
    QGridLayout *mainLayout = new QGridLayout;

    mainLayout->setColumnStretch(0, 1);
    mainLayout->setColumnStretch(3, 1);
    mainLayout->addWidget(renderArea, 0, 0, 1, 4);
    mainLayout->setRowMinimumHeight(1, 6);
    mainLayout->addWidget(resetTransitionFunctionButton, 4, 1, Qt::AlignCenter);
	mainLayout->addWidget(hapticLabel, 6, 1, Qt::AlignCenter);
	mainLayout->addWidget(zoomInButton, 7, 1, Qt::AlignCenter);
	mainLayout->addWidget(zoomOutButton, 7, 2, Qt::AlignCenter);
    setLayout(mainLayout);

	functionChangedSlot();

    setWindowTitle(tr("HapticInterface SettingsBox"));
}

void SettingsBoxWidget::resetDemandedSlot()
{
	emit resetFunction();
}

void SettingsBoxWidget::pointAddedSlot(double a_x, double a_y)
{
	functionData->SetValueOnPoint((unsigned short)(a_x * functionData->GetMaxPoint()), a_y * functionData->GetValueOfMaxPoint());
	renderArea->update();
}

void SettingsBoxWidget::functionChangedSlot()
{
	renderArea->update();
}

void SettingsBoxWidget::zoomInHapticSlot()
{
	emit zoomInHaptic();
}

void SettingsBoxWidget::zoomOutHapticSlot()
{
	emit zoomOutHaptic();
}