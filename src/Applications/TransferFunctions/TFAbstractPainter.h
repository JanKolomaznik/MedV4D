#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include <QtGui/QWidget>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>

#include <QtGui/QPainter>

#include <ui_TFAbstractPainter.h>

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractPainter: public QWidget{

	Q_OBJECT

public:

	TFColorMapPtr getView();

	void correctView();	//need to refresh view after this call

	bool changed();

protected:

	Ui::TFAbstractPainter* painter_;

	bool changed_;
	TFSize margin_;
	TFSize colorBarSize_;
	TFSize paintAreaWidth_, paintAreaHeight_;
	TFPaintingPoint* drawHelper_;

	TFColorMapPtr view_;

	TFAbstractPainter(QWidget* parent = NULL);
	virtual ~TFAbstractPainter();

	void paintBackground_(QPainter* drawer, QRect rect, bool hsv = false);

	TFPaintingPoint correctCoords_(const TFPaintingPoint &point);
	TFPaintingPoint correctCoords_(int x, int y);

	void addLine_(int x1, int y1, int x2, int y2);
	void addLine_(TFPaintingPoint begin, TFPaintingPoint end);

	virtual void correctView_() = 0;	//need to refresh view after this call

	virtual void paintEvent(QPaintEvent* e) = 0;
	virtual void mousePressEvent(QMouseEvent* e) = 0;
	virtual void mouseReleaseEvent(QMouseEvent* e) = 0;
	virtual void mouseMoveEvent(QMouseEvent* e) = 0;

	virtual void addPoint_(TFPaintingPoint point) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER