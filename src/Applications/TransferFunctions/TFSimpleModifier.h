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

	TFSimpleModifier(TFAbstractModifier::Type type, const TFSize& domain);
	~TFSimpleModifier();

	void mousePress(const TFSize& x, const TFSize& y, MouseButton button);
	void mouseRelease(const TFSize& x, const TFSize& y);
	void mouseMove(const TFSize& x, const TFSize& y);

private slots:

	void activeViewChanged(int index);
	void histogramCheck(bool enabled);

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
	TFPaintingPoint inputHelper_;
	bool leftMousePressed_;

	void addPoint_(const int& x, const int& y);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER