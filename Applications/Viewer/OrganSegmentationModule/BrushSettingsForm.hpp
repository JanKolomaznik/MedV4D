#ifndef BRUSHSETTINGSFORM_HPP
#define BRUSHSETTINGSFORM_HPP

#include <QWidget>

#include "OrganSegmentationController.hpp"

namespace Ui {
class BrushSettingsForm;
}

class BrushSettingsForm : public QWidget
{
	Q_OBJECT

public:
	explicit BrushSettingsForm(QWidget *parent = 0);
	~BrushSettingsForm();

	DrawingBrush
	brush() const;

signals:
	void
	updated();

private:
	Ui::BrushSettingsForm *ui;
};

#endif // BRUSHSETTINGSFORM_HPP
