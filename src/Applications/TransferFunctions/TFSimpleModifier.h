#ifndef TF_SIMPLE_MODIFIER
#define TF_SIMPLE_MODIFIER

#include <TFViewModifier.h>
#include <ui_TFSimpleModifier.h>

namespace M4D {
namespace GUI {

class TFSimpleModifier: public TFViewModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFSimpleModifier> Ptr;

	TFSimpleModifier(
		TFAbstractFunction<TF_DIMENSION_1>::Ptr function,
		TFSimplePainter::Ptr painter);

	~TFSimpleModifier();

protected slots:

	virtual void activeView_changed(int index);

protected:

	enum ActiveView{
		Active1,
		Active2,
		Active3,
		ActiveAlpha
	};	
	ActiveView activeView_;

	Ui::TFSimpleModifier* simpleTools_;
	QWidget* simpleWidget_;

	bool firstOnly_;

	bool leftMousePressed_;
	TF::PaintingPoint inputHelper_;

	virtual void createTools_();

	void computeInput_();
	std::vector<int> computeZoomMoveIncrements_(const int moveX, const int moveY);

	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);
	virtual void wheelEvent(QWheelEvent *e);

	virtual void addPoint_(const int x, const int y);
	void addLine_(TF::PaintingPoint begin, TF::PaintingPoint end);
	void addLine_(int x1, int y1, int x2, int y2);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER