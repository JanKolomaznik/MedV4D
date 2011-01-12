#ifndef TF_SIMPLE_MODIFIER
#define TF_SIMPLE_MODIFIER

#include <TFAbstractModifier.h>

namespace M4D {
namespace GUI {

class TFSimpleModifier: public TFAbstractModifier{

public:

	typedef boost::shared_ptr<TFSimpleModifier> Ptr;

	TFSimpleModifier(TFAbstractModifier::Type type);
	~TFSimpleModifier();

	void mousePress(TFSize x, TFSize y, MouseButton button);
	void mouseRelease(TFSize x, TFSize y);
	void mouseMove(TFSize x, TFSize y);

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

	void addPoint_(TFSize x, TFSize y);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_MODIFIER