#include <QtGui>
#include <sstream>
#include <string>

#include "transitionFunctionRenderArea.h"
#include "settingsBoxWidget.h"

const int IdRole = Qt::UserRole;

SettingsBoxWidget::SettingsBoxWidget(transitionFunction* functionData, QWidget* parent) : QWidget(parent)
{
	setBackgroundRole(QPalette::Base);
	setAutoFillBackground(true);
	
	this->functionData = functionData;
	
	renderArea = new transitionFunctionRenderAreaWidget(functionData);

	mouseCoordinatesLabelLabel = new QLabel(tr("Mouse coordinates (x, y):"));
	mouseCoordinatesLabel = new QLabel(tr("(0, 0)"));

	pointStyleCheckBox = new QCheckBox(tr("&Movable and deletable points"));
	pointStyleCheckBox->setCheckState(Qt::CheckState::Unchecked);
	
	resetTransitionFunctionButton = new QPushButton(tr("&Reset function"));

	zoomInButton = new QPushButton(tr("Zoom &in"));
	zoomOutButton = new QPushButton(tr("Zoom &out"));
	
	hapticLabel = new QLabel(tr("Haptic function:"));

	loadButton = new QPushButton(tr("&Load function"));
	saveButton = new QPushButton(tr("&Save function"));

	xLabel = new QLabel(tr("X:"));
	yLabel = new QLabel(tr("Y:"));
	
	xSpinBox = new QSpinBox();
	xSpinBox->setRange(functionData->GetMinPoint(), functionData->GetMaxPoint());
	xSpinBox->setSingleStep(1);

	ySpinBox = new QDoubleSpinBox();
	ySpinBox->setRange(functionData->GetValueOfMinPoint(), functionData->GetValueOfMaxPoint());
	ySpinBox->setSingleStep(0.01);

	addPointButton = new QPushButton(tr("Add new point"));
	setSolidFromButton = new QPushButton(tr("Set solid border"));
	unsetSolidFromButton = new QPushButton(tr("Unset solid border"));

    connect(resetTransitionFunctionButton, SIGNAL(clicked()), this, SLOT(resetDemandedSlot()));
	connect(zoomInButton, SIGNAL(clicked()), this, SLOT(zoomInHapticSlot()));
	connect(zoomOutButton, SIGNAL(clicked()), this, SLOT(zoomOutHapticSlot()));
	connect(renderArea, SIGNAL(addPointSignal(double, double)), this, SLOT(pointAddedSlot(double, double)));
	connect(renderArea, SIGNAL(mouseCoordinatesChangedSignal(double, double)), this, SLOT(mouseCoordinatesChangedSlot(double, double)));
	connect(pointStyleCheckBox, SIGNAL( stateChanged( int )), renderArea, SLOT( stateChangedSlot( int )));
	connect(loadButton, SIGNAL(clicked()), this, SLOT(loadFunctionSlot()));
	connect(saveButton, SIGNAL(clicked()), this, SLOT(saveFunctionSlot()));
	connect(addPointButton, SIGNAL(clicked()), this, SLOT(pointAddDemandSlot()));
	connect(setSolidFromButton, SIGNAL(clicked()), this, SLOT(setSolidFromDemandSlot()));
	connect(unsetSolidFromButton, SIGNAL(clicked()), this, SLOT(unsetSolidFromDemandSlot()));
	
    QGridLayout *mainLayout = new QGridLayout;

    mainLayout->setColumnStretch(0, 1);
    mainLayout->setColumnStretch(3, 1);
    mainLayout->addWidget(renderArea, 0, 0, 1, 4);
    mainLayout->setRowMinimumHeight(1, 6);
	mainLayout->addWidget(xLabel, 2, 1, Qt::AlignCenter);
	mainLayout->addWidget(xSpinBox, 2, 2, Qt::AlignCenter);
	mainLayout->addWidget(yLabel, 2, 3, Qt::AlignCenter);
	mainLayout->addWidget(ySpinBox, 2, 4, Qt::AlignCenter);
	mainLayout->addWidget(addPointButton, 3, 1, Qt::AlignCenter);
	mainLayout->addWidget(setSolidFromButton, 3, 2, Qt::AlignCenter);
	mainLayout->addWidget(unsetSolidFromButton, 3, 3, Qt::AlignCenter);
	mainLayout->addWidget(mouseCoordinatesLabelLabel, 4, 1, Qt::AlignCenter);
	mainLayout->addWidget(mouseCoordinatesLabel, 4, 2, Qt::AlignRight);
	mainLayout->addWidget(pointStyleCheckBox, 5, 1, Qt::AlignCenter);
    mainLayout->addWidget(resetTransitionFunctionButton, 6, 1, Qt::AlignCenter);
	mainLayout->addWidget(hapticLabel, 7, 1, Qt::AlignCenter);
	mainLayout->addWidget(zoomInButton, 8, 1, Qt::AlignCenter);
	mainLayout->addWidget(zoomOutButton, 8, 2, Qt::AlignCenter);
	mainLayout->addWidget(loadButton, 9, 1, Qt::AlignCenter);
	mainLayout->addWidget(saveButton, 9, 2, Qt::AlignCenter);
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
	xSpinBox->setRange(functionData->GetMinPoint(), functionData->GetMaxPoint());
	ySpinBox->setRange(functionData->GetValueOfMinPoint(), functionData->GetValueOfMaxPoint());
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

void SettingsBoxWidget::mouseCoordinatesChangedSlot( double a_x, double a_y )
{
	std::stringstream ss;
	ss << "( " << (unsigned short)(a_x * functionData->GetMaxPoint()) << ", " << a_y * functionData->GetValueOfMaxPoint() << " )";
	std::string s = ss.str();
	mouseCoordinatesLabel->setText(tr(s.c_str()));
}

void SettingsBoxWidget::closeEvent( QCloseEvent *event )
{
	event->ignore();
}

void SettingsBoxWidget::loadFunctionSlot()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Load transfer function"));
	if (!fileName.isEmpty())
	{
		functionData->LoadFromFile(fileName.toStdString());
	}
	renderArea->update();
}

void SettingsBoxWidget::saveFunctionSlot()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save transfer function"));
	if (!fileName.isEmpty())
	{
		functionData->SaveToFile(fileName.toStdString());
	}
}

void SettingsBoxWidget::pointAddDemandSlot()
{
	pointAddedSlot(((double)xSpinBox->value())/(double)(functionData->GetMaxPoint() - functionData->GetMinPoint()), ySpinBox->value());
	renderArea->update();
}

void SettingsBoxWidget::setSolidFromDemandSlot()
{
	functionData->SetSolidFrom(xSpinBox->value());
	renderArea->update();
}

void SettingsBoxWidget::unsetSolidFromDemandSlot()
{
	functionData->SetSolidFrom(-1);
	renderArea->update();
}