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

	TFSimpleModifier(TFAbstractModifier::Type type, const TFSize domain);
	~TFSimpleModifier();

	void mousePress(const int x, const int y, MouseButton button);
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

	TFAbstractModifier::Type type_;

	bool leftMousePressed_;
	TFPaintingPoint inputHelper_;

	bool zoomMovement_;
	TFPaintingPoint zoomMoveHelper_;

	void addPoint_(const int x, const int y);

	void updateZoomTools_();
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER