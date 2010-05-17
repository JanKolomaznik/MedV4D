#ifndef SETTINGS_BOX_WIDGET_H
#define SETTINGS_BOX_WIDGET_H

#include <QWidget>
#include "transitionFunction.h"

QT_BEGIN_NAMESPACE
class QPushButton;
class QLabel;
class QSpinBox;
class QDoubleSpinBox;
QT_END_NAMESPACE
class transitionFunctionRenderAreaWidget;

class SettingsBoxWidget : public QWidget
{
    Q_OBJECT

public:
    SettingsBoxWidget(transitionFunction* functionData, QWidget* parent = 0);

signals:
	void resetFunction();
	void zoomOutHaptic();
	void zoomInHaptic();

private slots:
	void pointAddedSlot(double a_x, double a_y);
	void resetDemandedSlot();
    void functionChangedSlot();
	void zoomInHapticSlot();
	void zoomOutHapticSlot();

private:
    transitionFunctionRenderAreaWidget *renderArea;
	transitionFunction* functionData;
	QLabel* hapticLabel;
	QPushButton* resetTransitionFunctionButton;
	QPushButton* zoomInButton;
	QPushButton* zoomOutButton;
};

#endif