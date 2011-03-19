#ifndef TF_PALETTE_BUTTON
#define TF_PALETTE_BUTTON

#include <TFCommon.h>
#include <QtGui/QWidget>
#include <QtGui/QAction>

namespace M4D {
namespace GUI {

class TFPaletteButton: public QWidget{

	Q_OBJECT

public:

	TFPaletteButton(QWidget* parent, const TF::Size index);
	~TFPaletteButton();

	void setup();

	void activate();
	void deactivate();

signals:

	void Triggered();

protected:

	void mouseReleaseEvent(QMouseEvent*);
	void paintEvent(QPaintEvent*);

private:

	TF::Size index_;
	bool active_;

	const TF::Size size_;

	void drawBorder_(QPainter* drawer, QColor color, int brushWidth);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_PALETTE_BUTTON