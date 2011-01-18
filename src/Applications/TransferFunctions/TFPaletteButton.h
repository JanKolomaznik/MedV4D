#ifndef TF_PALETTE_BUTTON
#define TF_PALETTE_BUTTON

#include <TFTypes.h>
#include <QtGui/QWidget>
#include <QtGui/QAction>

namespace M4D {
namespace GUI {

class TFPaletteButton: public QWidget{

	Q_OBJECT

public:

	TFPaletteButton(QWidget* parent, const TFSize& index);
	~TFPaletteButton();

	void activate();
	void deactivate();

signals:

	void Triggered();

protected:

	void mouseReleaseEvent(QMouseEvent*);
	void paintEvent(QPaintEvent*);

private:

	TFSize index_;
	bool active_;

	void drawBorder_(QPainter* drawer, QColor color, int brushWidth);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_PALETTE_BUTTON