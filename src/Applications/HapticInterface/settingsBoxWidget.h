#ifndef SETTINGS_BOX_WIDGET_H
#define SETTINGS_BOX_WIDGET_H

#include <QWidget>
#include "transitionFunction.h"

QT_BEGIN_NAMESPACE
class QPushButton;
class QLabel;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QCloseEvent;
QT_END_NAMESPACE
class transitionFunctionRenderAreaWidget;

class SettingsBoxWidget : public QWidget
{
    Q_OBJECT

public:
    SettingsBoxWidget(transitionFunction* functionData, QWidget* parent = 0);
	void setCloseEnabled();

public slots:
	void functionChangedSlot();

signals:
	void resetFunction();
	void zoomOutHaptic();
	void zoomInHaptic();
	void setTraceLogOn( std::string file );
	void setTraceLogOff();
	void setToOriginal();

private slots:
	void pointAddedSlot(double a_x, double a_y);
	void mouseCoordinatesChangedSlot(double a_x, double a_y);
	void pointAddDemandSlot();
	void setSolidFromDemandSlot();
	void unsetSolidFromDemandSlot();
	void setSolidToDemandSlot();
	void unsetSolidToDemandSlot();
	void resetDemandedSlot();
	void zoomInHapticSlot();
	void zoomOutHapticSlot();
	void loadFunctionSlot();
	void saveFunctionSlot();
	void setTraceLogOnOffSlot();
	void setToOriginalSlot();

private:
    transitionFunctionRenderAreaWidget *renderArea;
	transitionFunction* functionData;
	QLabel* hapticLabel;
	QLabel* mouseCoordinatesLabelLabel;
	QLabel* mouseCoordinatesLabel;
	QLabel* xLabel;
	QLabel* yLabel;
	QCheckBox* pointStyleCheckBox;
	QPushButton* resetTransitionFunctionButton;
	QPushButton* zoomInButton;
	QPushButton* zoomOutButton;
	QPushButton* loadButton;
	QPushButton* saveButton;
	QPushButton* addPointButton;
	QPushButton* setSolidFromButton;
	QPushButton* unsetSolidFromButton;
	QPushButton* setSolidToButton;
	QPushButton* unsetSolidToButton;
	QPushButton* setTraceLogOnOffButton;
	QPushButton* setToOriginalButton;
	QSpinBox* xSpinBox;
	QDoubleSpinBox* ySpinBox;
	void closeEvent(QCloseEvent *event);
	bool traceLogState;
	bool closeEnabled;
};

#endif
