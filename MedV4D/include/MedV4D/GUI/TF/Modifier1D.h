#ifndef TF_MODIFIER_1D
#define TF_MODIFIER_1D

#include "MedV4D/GUI/TF/ViewModifier.h"

#include "MedV4D/GUI/TF/Painter1D.h"

#include "MedV4D/generated/ui_Modifier1D.h"

namespace M4D {
namespace GUI {

class Modifier1D: public ViewModifier{

	Q_OBJECT

public:

	typedef std::shared_ptr<Modifier1D> Ptr;

	Modifier1D(
		TransferFunctionInterface::Ptr function,
		Painter1D::Ptr painter);

	~Modifier1D();

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

	Ui::Modifier1D* simpleTools_;
	QWidget* simpleWidget_;

	bool firstOnly_;

	bool leftMousePressed_;
	TF::PaintingPoint inputHelper_;

	virtual void createTools_();

	void computeInput_();
	std::vector<int> computeZoomMoveIncrements_(const int moveX, const int moveY);

	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);

	virtual void addPoint_(const int x, const int y);
	void addLine_(TF::PaintingPoint begin, TF::PaintingPoint end);
	void addLine_(int x1, int y1, int x2, int y2);
};

} // namespace GUI
} // namespace M4D

#endif //TF_MODIFIER_1D
