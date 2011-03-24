#ifndef TF_ABSTRACT_MODIFIER
#define TF_ABSTRACT_MODIFIER

#include <TFCommon.h>
#include <TFWorkCopy.h>

#include <QtGui/QWidget>

namespace M4D {
namespace GUI {

class RefreshSignal: public QObject{

	Q_OBJECT

signals:

	void RefreshView();

protected:
	RefreshSignal(){}
	virtual ~RefreshSignal(){}
};

template<TF::Size dim>
class TFAbstractModifier: public RefreshSignal{

public:

	typedef typename boost::shared_ptr<TFAbstractModifier<dim>> Ptr;

	QWidget* getTools(){

		return toolsWidget_;
	}

	typename TFWorkCopy<dim>::Ptr getWorkCopy() const{

		return workCopy_;
	}

	void setInputArea(QRect inputArea){

		inputArea_ = inputArea;
		workCopy_->resize(inputArea_.width(), inputArea_.height());
	}

	bool changed(){

		if(changed_)
		{
			changed_ = false;
			return true;
		}
		return false;
	}

	virtual void mousePress(const int x, const int y, Qt::MouseButton button){}
	virtual void mouseRelease(const int x, const int y){}
	virtual void mouseMove(const int x, const int y){}
	virtual void mouseWheel(const int steps, const int x, const int y){}
	virtual void keyPressed(QKeySequence keySequence){}
/*
signals:

	void RefreshView();
*/
protected:

	QWidget* toolsWidget_;

	bool changed_;

	typename TFWorkCopy<dim>::Ptr workCopy_;
	QRect inputArea_;
	const TF::PaintingPoint ignorePoint_;

	TFAbstractModifier():
		ignorePoint_(-1, -1),
		toolsWidget_(NULL),
		changed_(true){
	}

	virtual ~TFAbstractModifier(){}

	virtual void addPoint_(const int x, const int y) = 0;

	TF::PaintingPoint getRelativePoint_(const int x, const int y, bool acceptOutOfBounds = false){	

		int xMax = inputArea_.width() - 1;
		int yMax = inputArea_.height();
		
		TF::PaintingPoint corrected = TF::PaintingPoint(x - inputArea_.x(), inputArea_.height() - (y - inputArea_.y()));

		bool outOfBounds = false;
		if( corrected.x < 0 ||
			corrected.x > xMax ||
			corrected.y < 0 ||
			corrected.y > yMax)
		{
			outOfBounds = true;
		}	
		if(outOfBounds && !acceptOutOfBounds) return ignorePoint_;
		return corrected;
	}

	void addLine_(TF::PaintingPoint begin, TF::PaintingPoint end){
		
		addLine_(begin.x, begin.y, end.x, end.y);
	}

	void addLine_(int x1, int y1, int x2, int y2){ // assumes x1<x2, |y2-y1|<|x2-x1|
		
		//tfAssert((x1 < x2) && (abs(y2-y1) < abs(x2-x1)));
		if(x1==x2 && y1==y2) addPoint_(x1,y1);

		int D, ax, ay, sx, sy;

		sx = x2 - x1;
		ax = abs( sx ) << 1;

		if ( sx < 0 ) sx = -1;
		else if ( sx > 0 ) sx = 1;

		sy = y2 - y1;
		ay = abs( sy ) << 1;

		if ( sy < 0 ) sy = -1;
		else if ( sy > 0 ) sy = 1;

		if ( ax > ay )                          // x coordinate is dominant
		{
			D = ay - (ax >> 1);                   // initial D
			ax = ay - ax;                         // ay = increment0; ax = increment1

			while ( x1 != x2 )
			{
				addPoint_(x1,y1);
				if ( D >= 0 )                       // lift up the Y coordinate
				{
					y1 += sy;
					D += ax;
				}
				else
				{
					D += ay;
				}
				x1 += sx;
			}
		}
		else                                    // y coordinate is dominant
		{
			D = ax - (ay >> 1);                   // initial D
			ay = ax - ay;                         // ax = increment0; ay = increment1

			while ( y1 != y2 )
			{
				addPoint_(x1,y1);
				if ( D >= 0 )                       // lift up the X coordinate
				{
					x1 += sx;
					D += ay;
				}
				else
				{
					D += ax;
				}
				y1 += sy;
			}
		}
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_MODIFIER