#ifndef TF_SIMPLE_MODIFIER
#define TF_SIMPLE_MODIFIER

#include <TFAbstractModifier.h>

namespace M4D {
namespace GUI {

class TFSimpleModifier: public TFAbstractModifier{

public:

	typedef boost::shared_ptr<TFSimpleModifier> Ptr;

	TFSimpleModifier(TFAbstractModifier::Type type, const TFSize& domain);
	~TFSimpleModifier();

	void mousePress(const TFSize& x, const TFSize& y, MouseButton button);
	void mouseRelease(const TFSize& x, const TFSize& y);
	void mouseMove(const TFSize& x, const TFSize& y);

private:

	enum ActiveView{
		Active1,
		Active2,
		Active3,
		ActiveAlpha
	};	
	ActiveView activeView_;

	ActiveView active1Next_();
	ActiveView active2Next_();
	ActiveView active3Next_();
	ActiveView activeAlphaNext_();

	TFAbstractModifier::Type type_;
	TFPaintingPoint inputHelper_;
	bool leftMousePressed_;

	void addPoint_(const TFSize& x, const TFSize& y);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER