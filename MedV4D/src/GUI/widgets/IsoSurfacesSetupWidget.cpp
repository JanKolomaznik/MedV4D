#include "MedV4D/GUI/widgets/IsoSurfacesSetupWidget.h"
#include "MedV4D/generated/ui_IsoSurfacesSetupWidget.h"

#include "MedV4D/GUI/utils/QtM4DTools.h"

IsoSurfacesSetupWidget::IsoSurfacesSetupWidget(QWidget *parent) :
QWidget(parent),
ui(new Ui::IsoSurfacesSetupWidget)
{
	ui->setupUi(this);

	mLayout = new QGridLayout();
	ui->mScrollAreaIsoSurfaceSettings->widget()->setLayout(mLayout);

	addIsoSurface();
}

IsoSurfacesSetupWidget::~IsoSurfacesSetupWidget()
{
	delete ui;
}

vorgl::IsoSurfaceDefinitionList
IsoSurfacesSetupWidget::isoSurfaces() const
{
	vorgl::IsoSurfaceDefinitionList surfaces;
	surfaces.resize(mIsoSurfaces.size());
	for (size_t i = 0; i < mIsoSurfaces.size(); ++i) {
		surfaces[i] = mIsoSurfaces[i]->isoSurfaceDefinition();
	}

	return surfaces;
}

void
IsoSurfacesSetupWidget::setIsoSurfaces(const vorgl::IsoSurfaceDefinitionList &aIsoSurfaces)
{
	ui->mAddNewIsoSurfaceButton->setEnabled(true);
	M4D::GUI::QtSignalBlocker blocker(this);
	int countDifference = int(aIsoSurfaces.size()) - int(mIsoSurfaces.size());
	for (int i = 0; i < countDifference; ++i) {
		addIsoSurface();
	}
	for (int i = 0; i < (-1 * countDifference); ++i) {
		removeIsoSurface();
	}
	for (size_t i = 0; i < mIsoSurfaces.size(); ++i) {
		M4D::GUI::QtSignalBlocker blocker(mIsoSurfaces[i].get());
		mIsoSurfaces[i]->setIsoSurfaceDefinition(aIsoSurfaces[i]);
	}
}

void IsoSurfacesSetupWidget::addIsoSurface()
{
	auto widget = std::unique_ptr<SingleIsoSurfaceSetupWidget>(new SingleIsoSurfaceSetupWidget());


	mLayout->addWidget(widget.get(), mLayout->rowCount(), 0, 1, 1);
	QObject::connect(
			widget.get(),
			&SingleIsoSurfaceSetupWidget::updated,
			this,
			&IsoSurfacesSetupWidget::isoSurfaceSetupUpdated,
			Qt::QueuedConnection);
	mIsoSurfaces.push_back(std::move(widget));

	if (mIsoSurfaces.size() >= cMaxIsoSurfaceCount) {
		ui->mAddNewIsoSurfaceButton->setEnabled(false);
	}
	emit isoSurfaceSetupUpdated();
}

void IsoSurfacesSetupWidget::removeIsoSurface()
{
	if (mIsoSurfaces.empty()) {
		return;
	}
	mIsoSurfaces.pop_back();
	if (mIsoSurfaces.size() < cMaxIsoSurfaceCount) {
		ui->mAddNewIsoSurfaceButton->setEnabled(true);
	}
	emit isoSurfaceSetupUpdated();
}
