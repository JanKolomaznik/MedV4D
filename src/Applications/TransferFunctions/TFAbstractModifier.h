#ifndef TF_ABSTRACT_MODIFIER
#define TF_ABSTRACT_MODIFIER

#include <TFTypes.h>
#include <TFWorkCopy.h>

namespace M4D {
namespace GUI {

class TFAbstractModifier{

public:

	typedef boost::shared_ptr<TFAbstractModifier> Ptr;

	enum Type{
		TFModifierGrayscale,
		TFModifierGrayscaleAlpha,
		TFModifierRGB,
		TFModifierRGBa,
		TFModifierHSV,
		TFModifierHSVa
	};

	TFWorkCopy::Ptr getWorkCopy();
	//void setWorkCopy(TFWorkCopy::Ptr workCopy);
	void setInputArea(TFArea inputArea);

	virtual void mousePress(const TFSize& x, const TFSize& y, MouseButton button) = 0;
	virtual void mouseRelease(const TFSize& x, const TFSize& y) = 0;
	virtual void mouseMove(const TFSize& x, const TFSize& y) = 0;

	M4D::Common::TimeStamp getLastChangeTime();

protected:

	M4D::Common::TimeStamp lastChange_;

	TFWorkCopy::Ptr workCopy_;
	TFArea inputArea_;

	TFAbstractModifier();
	virtual ~TFAbstractModifier();

	void addLine_(int x1, int y1, int x2, int y2);
	void addLine_(TFPaintingPoint begin, TFPaintingPoint end);

	TFPaintingPoint getRelativePoint_(const TFSize& x, const TFSize& y);

	virtual void addPoint_(const int& x, const int& y) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_MODIFIER