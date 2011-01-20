#ifndef TF_POLYGON_MODIFIER
#define TF_POLYGON_MODIFIER

#include <TFAbstractModifier.h>

namespace M4D {
namespace GUI {

class TFPolygonModifier: public TFAbstractModifier{

public:

	typedef boost::shared_ptr<TFPolygonModifier> Ptr;

	TFPolygonModifier(TFAbstractModifier::Type type, const TFSize& domain);
	~TFPolygonModifier();

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

	const TFSize baseRadius_;
	const TFSize topRadius_;

	void addPolygon_(const int& x, const int& y);
	void addPoint_(const int& x, const int& y);
};

} // namespace GUI
} // namespace M4D

#endif //TF_POLYGON_MODIFIER