#include "MedV4D/GUI/widgets/SingleIsoSurfaceSetupWidget.h"
#include "MedV4D/generated/ui_SingleIsoSurfaceSetupWidget.h"

#include <QColor>

#include "MedV4D/GUI/utils/QtM4DTools.h"

SingleIsoSurfaceSetupWidget::SingleIsoSurfaceSetupWidget(QWidget *parent) :
QWidget(parent),
ui(new Ui::SingleIsoSurfaceSetupWidget)
{
	ui->setupUi(this);
	ui->mSurfaceColorSetter->enableAlpha(true);

	QObject::connect(ui->mIsoValueSetter, &DoubleSpinBoxWithSlider::valueChanged, this, &SingleIsoSurfaceSetupWidget::updated);
	QObject::connect(ui->mSurfaceColorSetter, &ColorChooserButton::colorUpdated, this, &SingleIsoSurfaceSetupWidget::updated);
}

SingleIsoSurfaceSetupWidget::~SingleIsoSurfaceSetupWidget()
{
	delete ui;
}

vorgl::IsoSurfaceDefinition
SingleIsoSurfaceSetupWidget::isoSurfaceDefinition() const
{
	vorgl::IsoSurfaceDefinition isoSurface;
	isoSurface.isoValue = ui->mIsoValueSetter->value();
	isoSurface.isoSurfaceColor = glm::vec4(
			ui->mSurfaceColorSetter->color().redF(),
			ui->mSurfaceColorSetter->color().greenF(),
			ui->mSurfaceColorSetter->color().blueF(),
			ui->mSurfaceColorSetter->color().alphaF()
			);
	return isoSurface;
}

void
SingleIsoSurfaceSetupWidget::setIsoSurfaceDefinition(const vorgl::IsoSurfaceDefinition &aSurfaceDefinition)
{
	M4D::GUI::QtSignalBlocker blocker1(ui->mIsoValueSetter);
	M4D::GUI::QtSignalBlocker blocker2(ui->mSurfaceColorSetter);
	if (ui->mIsoValueSetter->value() != aSurfaceDefinition.isoValue) {
		ui->mIsoValueSetter->setValue(aSurfaceDefinition.isoValue);
	}
	auto color = QColor::fromRgbF(
		aSurfaceDefinition.isoSurfaceColor[0],
		aSurfaceDefinition.isoSurfaceColor[1],
		aSurfaceDefinition.isoSurfaceColor[2],
		aSurfaceDefinition.isoSurfaceColor[3]
		);
	if (color != ui->mSurfaceColorSetter->color()) {
		ui->mSurfaceColorSetter->setColor(color);
	}
}
