#ifndef TF_SIMPLE_MODIFIER
#define TF_SIMPLE_MODIFIER

#include <TFAbstractModifier.h>
#include <ui_TFSimpleModifier.h>

namespace M4D {
namespace GUI {

class TFSimpleModifier: public TFAbstractModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFSimpleModifier> Ptr;

	enum Mode{
		Grayscale,
		RGB,
		HSV
	};

	TFSimpleModifier(TFWorkCopy::Ptr workCopy, Mode mode, bool alpha);
	~TFSimpleModifier();

	void mousePress(const int x, const int y, Qt::MouseButton button);
	void mouseRelease(const int x, const int y);
	void mouseMove(const int x, const int y);
	void mouseWheel(const int steps, const int x, const int y);

private slots:

	void activeView_changed(int index);
	void histogram_check(bool enabled);

	void maxZoomSpin_changed(int value);

private:

	enum ActiveView{
		Active1,
		Active2,
		Active3,
		ActiveAlpha
	};	
	ActiveView activeView_;

	Ui::TFSimpleModifier* tools_;

	Mode mode_;
	bool alpha_;

	bool leftMousePressed_;
	TF::PaintingPoint inputHelper_;

	bool zoomMovement_;
	TF::PaintingPoint zoomMoveHelper_;

	void addPoint_(const int x, const int y);

	void updateZoomTools_();
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER