#ifndef TF_ABSTRACT_MODIFIER
#define TF_ABSTRACT_MODIFIER

#include <TFCommon.h>
#include <TFWorkCopy.h>

#include <QtGui/QWidget>

namespace M4D {
namespace GUI {

class TFAbstractModifier: public QWidget{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFAbstractModifier> Ptr;

	QWidget* getTools();

	TFWorkCopy::Ptr getWorkCopy() const;
	//void setWorkCopy(TFWorkCopy::Ptr workCopy);
	void setInputArea(QRect inputArea);

	M4D::Common::TimeStamp getLastChangeTime();

	virtual void mousePress(const int x, const int y, Qt::MouseButton button){}
	virtual void mouseRelease(const int x, const int y){}
	virtual void mouseMove(const int x, const int y){}
	virtual void mouseWheel(const int steps, const int x, const int y){}
	virtual void keyPressed(QKeySequence keySequence){}

signals:

	void RefreshView();

protected:

	QWidget* toolsWidget_;

	M4D::Common::TimeStamp lastChange_;

	TFWorkCopy::Ptr workCopy_;
	QRect inputArea_;
	const TF::PaintingPoint ignorePoint_;

	TFAbstractModifier();
	virtual ~TFAbstractModifier();

	void addLine_(int x1, int y1, int x2, int y2);
	void addLine_(TF::PaintingPoint begin, TF::PaintingPoint end);

	TF::PaintingPoint getRelativePoint_(const int x, const int y, bool acceptOutOfBounds = false);

	virtual void addPoint_(const int x, const int y) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_MODIFIER