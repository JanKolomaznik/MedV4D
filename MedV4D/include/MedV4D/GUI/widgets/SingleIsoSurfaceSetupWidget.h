#ifndef SINGLEISOSURFACESETUPWIDGET_H
#define SINGLEISOSURFACESETUPWIDGET_H

#include <QWidget>
#include <vorgl/VolumeRenderer.hpp>


namespace Ui {
class SingleIsoSurfaceSetupWidget;
}

class SingleIsoSurfaceSetupWidget : public QWidget
{
	Q_OBJECT

public:
	explicit SingleIsoSurfaceSetupWidget(QWidget *parent = 0);
	~SingleIsoSurfaceSetupWidget();

	vorgl::IsoSurfaceDefinition
	isoSurfaceDefinition() const;

	void
	setIsoSurfaceDefinition(const vorgl::IsoSurfaceDefinition &aSurfaceDefinition);

signals:
	void
	updated();

private:
	Ui::SingleIsoSurfaceSetupWidget *ui;
};

#endif // SINGLEISOSURFACESETUPWIDGET_H
