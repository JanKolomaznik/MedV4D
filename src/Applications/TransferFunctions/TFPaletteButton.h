#ifndef TF_PALETTE_BUTTON
#define TF_PALETTE_BUTTON

#include <TFCommon.h>

#include <QtGui/QWidget>
#include <QtGui/QAction>
#include <QtGui/QPainter>

namespace M4D {
namespace GUI {

class TFPaletteButton: public QWidget{

	Q_OBJECT

public:

	TFPaletteButton(QWidget* parent, const TF::Size index);
	~TFPaletteButton();

	void setup();
	void setPreview(const QPixmap& preview);

	void activate();
	void deactivate();

signals:

	void Triggered(TF::Size index);

protected:

	void mouseReleaseEvent(QMouseEvent*);
	void paintEvent(QPaintEvent*);

private:

	TF::Size index_;
	QPixmap preview_;
	const TF::Size size_;
	
	bool active_;

	QPixmap activePreview_;
};

} // namespace GUI
} // namespace M4D

#endif	//TF_PALETTE_BUTTON