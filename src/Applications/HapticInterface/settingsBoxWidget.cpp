#include <QtGui>
#include <sstream>
#include <string>

#include "transitionFunctionRenderArea.h"
#include "settingsBoxWidget.h"

const int IdRole = Qt::UserRole;

SettingsBoxWidget::SettingsBoxWidget(transitionFunction* functionData, QWidget* parent) : QWidget(parent)
{
	traceLogState = false;
	closeEnabled = false;
	
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
	setSolidFromButton = new QPushButton(tr("Set solid from border"));
	unsetSolidFromButton = new QPushButton(tr("Unset solid from border"));
	setSolidToButton = new QPushButton(tr("Set solid to border"));
	unsetSolidToButton = new QPushButton(tr("Unset solid to border"));
	setTraceLogOnOffButton = new QPushButton(tr("Start trace log"));

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
	connect(setSolidToButton, SIGNAL(clicked()), this, SLOT(setSolidToDemandSlot()));
	connect(unsetSolidToButton, SIGNAL(clicked()), this, SLOT(unsetSolidToDemandSlot()));
	connect(setTraceLogOnOffButton, SIGNAL(clicked()), this, SLOT(setTraceLogOnOffSlot()));
	
    QGridLayout *mainLayout = new QGridLayout;

    mainLayout->setColumnStretch(0, 1);
    mainLayout->setColumnStretch(3, 1);
    mainLayout->addWidget(renderArea, 0, 0, 1, 5);
    mainLayout->setRowMinimumHeight(1, 6);
	mainLayout->addWidget(xLabel, 1, 0, Qt::AlignCenter);
	mainLayout->addWidget(xSpinBox, 1, 1, Qt::AlignCenter);
	mainLayout->addWidget(yLabel, 1, 2, Qt::AlignCenter);
	mainLayout->addWidget(ySpinBox, 1, 3, Qt::AlignCenter);
	mainLayout->addWidget(addPointButton, 3, 0, Qt::AlignCenter);
	mainLayout->addWidget(setSolidFromButton, 3, 2, Qt::AlignCenter);
	mainLayout->addWidget(unsetSolidFromButton, 3, 4, Qt::AlignCenter);
	mainLayout->addWidget(setSolidToButton, 3, 1, Qt::AlignCenter);
	mainLayout->addWidget(unsetSolidToButton, 3, 3, Qt::AlignCenter);
	mainLayout->addWidget(mouseCoordinatesLabelLabel, 4, 0, Qt::AlignCenter);
	mainLayout->addWidget(mouseCoordinatesLabel, 4, 1, Qt::AlignRight);
	mainLayout->addWidget(pointStyleCheckBox, 5, 0, Qt::AlignCenter);
    mainLayout->addWidget(resetTransitionFunctionButton, 6, 0, Qt::AlignCenter);
	mainLayout->addWidget(setTraceLogOnOffButton, 6, 1, Qt::AlignCenter);
	mainLayout->addWidget(hapticLabel, 7, 0, Qt::AlignCenter);
	mainLayout->addWidget(zoomInButton, 8, 0, Qt::AlignCenter);
	mainLayout->addWidget(zoomOutButton, 8, 1, Qt::AlignCenter);
	mainLayout->addWidget(loadButton, 9, 0, Qt::AlignCenter);
	mainLayout->addWidget(saveButton, 9, 1, Qt::AlignCenter);
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
	if (!closeEnabled)
	{
		event->ignore();
	}
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
	if (functionData->GetSolidTo() >= xSpinBox->value())
	{
		return;
	}
	functionData->SetSolidFrom(xSpinBox->value());
	renderArea->update();
}

void SettingsBoxWidget::unsetSolidFromDemandSlot()
{
	functionData->SetSolidFrom(-1);
	renderArea->update();
}

void SettingsBoxWidget::setSolidToDemandSlot()
{
	if ((functionData->GetSolidFrom() != -1) && (functionData->GetSolidFrom() <= xSpinBox->value()))
	{
		return;
	}
	functionData->SetSolidTo(xSpinBox->value());
	renderArea->update();
}

void SettingsBoxWidget::unsetSolidToDemandSlot()
{
	functionData->SetSolidTo(-1);
	renderArea->update();
}

void SettingsBoxWidget::setTraceLogOnOffSlot()
{
	if (traceLogState)
	{
		traceLogState = false;
		setTraceLogOnOffButton->setText(tr("Start trace log"));
		emit setTraceLogOff();
	}
	else
	{
		QString fileName = QFileDialog::getSaveFileName(this, tr("Save trace log"));
		if (!fileName.isEmpty())
		{
			traceLogState = true;
			setTraceLogOnOffButton->setText(tr("Stop trace log"));
			emit setTraceLogOn(fileName.toStdString());
		}
	}
}

void SettingsBoxWidget::setCloseEnabled()
{
	closeEnabled = true;
}