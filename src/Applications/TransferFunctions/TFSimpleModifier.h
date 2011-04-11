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

	typedef TFViewModifier::WorkCopy WorkCopy;

	enum Mode{
		Grayscale,
		RGB,
		HSV
	};

	TFSimpleModifier(WorkCopy::Ptr workCopy, Mode mode, bool alpha);
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

	Mode mode_;
	bool alpha_;

	bool leftMousePressed_;
	TF::PaintingPoint inputHelper_;

	virtual void createTools_();

	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);

	virtual void addPoint_(const int x, const int y);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER