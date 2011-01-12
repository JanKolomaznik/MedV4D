#ifndef TF_ABSTRACT_MODIFIER
#define TF_ABSTRACT_MODIFIER

#include <TFTypes.h>

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

	TFAbstractModifier();
	virtual ~TFAbstractModifier();

	TFColorMapPtr getWorkCopy();
	void setWorkCopy(TFColorMapPtr workCopy);
	void setInputArea(TFArea inputArea);

	virtual void mousePress(TFSize x, TFSize y, MouseButton button) = 0;
	virtual void mouseRelease(TFSize x, TFSize y) = 0;
	virtual void mouseMove(TFSize x, TFSize y) = 0;

	M4D::Common::TimeStamp getLastChangeTime();

protected:

	M4D::Common::TimeStamp lastChange_;

	TFColorMapPtr workCopy_;
	TFArea inputArea_;

	void addLine_(int x1, int y1, int x2, int y2);
	void addLine_(TFPaintingPoint begin, TFPaintingPoint end);

	TFPaintingPoint getRelativePoint_(TFSize x, TFSize y);

	virtual void addPoint_(TFSize x, TFSize y) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_MODIFIER