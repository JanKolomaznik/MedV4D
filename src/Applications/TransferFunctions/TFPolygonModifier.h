#ifndef TF_POLYGON_MODIFIER
#define TF_POLYGON_MODIFIER

#include <TFAbstractModifier.h>
#include <ui_TFPolygonModifier.h>

namespace M4D {
namespace GUI {

class TFPolygonModifier: public TFAbstractModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFPolygonModifier> Ptr;

	TFPolygonModifier(TFAbstractModifier::Type type, const TFSize& domain);
	~TFPolygonModifier();

	void mousePress(const TFSize& x, const TFSize& y, MouseButton button);
	void mouseRelease(const TFSize& x, const TFSize& y);
	void mouseMove(const TFSize& x, const TFSize& y);

private slots:

	void activeViewChanged(int index);
	void histogramCheck(bool enabled);
	void bottomSpinChanged(int value);
	void topSpinChanged(int value);

private:

	enum ActiveView{
		Active1,
		Active2,
		Active3,
		ActiveAlpha
	};	
	ActiveView activeView_;
	
	Ui::TFPolygonModifier* tools_;

	void active1Next_();
	void active2Next_();
	void active3Next_();
	void activeAlphaNext_();

	TFAbstractModifier::Type type_;
	TFPoint<TFSize,TFSize> inputHelper_;
	bool leftMousePressed_;

	TFSize baseRadius_;
	TFSize topRadius_;

	void addPolygon_(const int& x, const int& y);
	void addPoint_(const int& x, const int& y);
};

} // namespace GUI
} // namespace M4D

#endif //TF_POLYGON_MODIFIER