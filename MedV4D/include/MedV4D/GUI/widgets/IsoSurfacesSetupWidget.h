#ifndef ISOSURFACESSETUPWIDGET_H
#define ISOSURFACESSETUPWIDGET_H

#include <memory>

#include <QWidget>
#include <QGridLayout>

#include "MedV4D/GUI/widgets/SingleIsoSurfaceSetupWidget.h"

#include <vorgl/VolumeRenderer.hpp>

namespace Ui {
class IsoSurfacesSetupWidget;
}

class IsoSurfacesSetupWidget : public QWidget
{
	Q_OBJECT

public:
	static const int cMaxIsoSurfaceCount = 10;
	explicit IsoSurfacesSetupWidget(QWidget *parent = 0);
	~IsoSurfacesSetupWidget();

	vorgl::IsoSurfaceDefinitionList
	isoSurfaces() const;

	void
	setIsoSurfaces(const vorgl::IsoSurfaceDefinitionList &aIsoSurfaces);

public slots:
	void
	addIsoSurface();

	void
	removeIsoSurface();
signals:
	void
	isoSurfaceSetupUpdated();

private:
	Ui::IsoSurfacesSetupWidget *ui;

	QGridLayout *mLayout;

	std::vector<std::unique_ptr<SingleIsoSurfaceSetupWidget>> mIsoSurfaces;
};

#endif // ISOSURFACESSETUPWIDGET_H
