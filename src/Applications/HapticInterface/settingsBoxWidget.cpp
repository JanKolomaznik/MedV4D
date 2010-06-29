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

    connect(resetTransitionFunctionButton, SIGNAL(clicked()), this, SLOT(resetDemandedSlot()));
	connect(zoomInButton, SIGNAL(clicked()), this, SLOT(zoomInHapticSlot()));
	connect(zoomOutButton, SIGNAL(clicked()), this, SLOT(zoomOutHapticSlot()));
	connect(renderArea, SIGNAL(addPointSignal(double, double)), this, SLOT(pointAddedSlot(double, double)));
	connect(renderArea, SIGNAL(mouseCoordinatesChangedSignal(double, double)), this, SLOT(mouseCoordinatesChangedSlot(double, double)));
	connect(pointStyleCheckBox, SIGNAL( stateChanged( int )), renderArea, SLOT( stateChangedSlot( int )));
	
    QGridLayout *mainLayout = new QGridLayout;

    mainLayout->setColumnStretch(0, 1);
    mainLayout->setColumnStretch(3, 1);
    mainLayout->addWidget(renderArea, 0, 0, 1, 4);
    mainLayout->setRowMinimumHeight(1, 6);
	mainLayout->addWidget(mouseCoordinatesLabelLabel, 4, 1, Qt::AlignCenter);
	mainLayout->addWidget(mouseCoordinatesLabel, 4, 2, Qt::AlignRight);
	mainLayout->addWidget(pointStyleCheckBox, 5, 1, Qt::AlignCenter);
    mainLayout->addWidget(resetTransitionFunctionButton, 6, 1, Qt::AlignCenter);
	mainLayout->addWidget(hapticLabel, 7, 1, Qt::AlignCenter);
	mainLayout->addWidget(zoomInButton, 8, 1, Qt::AlignCenter);
	mainLayout->addWidget(zoomOutButton, 8, 2, Qt::AlignCenter);
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