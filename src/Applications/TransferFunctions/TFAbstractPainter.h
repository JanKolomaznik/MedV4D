#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include "Imaging/Histogram.h"

#include <TFTypes.h>
#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {

class TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFAbstractPainter> Ptr;

	TFAbstractPainter():histogramEnabled_(false){}
	virtual ~TFAbstractPainter(){};

	virtual void setArea(TFArea area) = 0;
	virtual TFArea getInputArea() = 0;

	virtual void drawBackground(QPainter* drawer) = 0;
	virtual void drawData(QPainter* drawer, TFColorMapPtr workCopy) = 0;
	void setHistogram(TFColorMapPtr histogram){ histogram_ = histogram; }
	void drawHistogram(bool enabled){ histogramEnabled_ = true; }

protected:

	TFArea area_;
	TFColorMapPtr histogram_;
	bool histogramEnabled_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER