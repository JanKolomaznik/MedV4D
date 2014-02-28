#ifndef TF_PALETTE_BUTTON
#define TF_PALETTE_BUTTON

#include "MedV4D/GUI/TF/Common.h"

#include <QtWidgets/QFrame>
#include <QPainter>
#include <QImage>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>

namespace M4D {
namespace GUI {

class PaletteButton: public QFrame{

	Q_OBJECT

public:
	static const TF::Size previewWidth = 128;
	static const TF::Size previewHeight = 128;

	PaletteButton(const TF::Size index, QWidget* parent = 0);
	~PaletteButton();

	virtual void setup(const std::string& name = "", bool enablePreview = true);
	void setName(const std::string& name);
	void setPreview(const QImage& preview);
	QImage getPreview();

	virtual void togglePreview(bool enabled);

	virtual void setActive(const bool active);
	virtual void setAvailable(const bool available);

	bool isActive();
	bool isAvailable();

signals:

	void Triggered(TF::Size index);

protected:

	static const TF::Size nameHeight_ = 25;
	static const TF::Size frameLineWidth_ = 2;

	TF::Size index_;
	QImage preview_;
	QLabel name_;
	QRect previewRect_;

	bool active_;
	bool available_;
	bool previewEnabled_;

	QPixmap activePreview_;
	QPixmap availablePreview_;

	void mouseReleaseEvent(QMouseEvent*);
	void paintEvent(QPaintEvent*);
};

class PaletteCheckButton: public PaletteButton{

	Q_OBJECT

public:

	PaletteCheckButton(const TF::Size index, QWidget* parent = 0);
	~PaletteCheckButton();

	void setup(const std::string& name = "", bool enablePreview = true);

	void togglePreview(bool enabled);

	void setActive(const bool active);
	void setAvailable(const bool available);

private slots:

	void check_toggled(bool toggled);

private:

	static const TF::Size checkWidth_ = 15;
	static const TF::Size checkIndent_ = 5;

	QCheckBox check_;
};

} // namespace GUI
} // namespace M4D

#endif	//TF_PALETTE_BUTTON
