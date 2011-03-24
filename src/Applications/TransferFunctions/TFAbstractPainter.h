#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include "Imaging/Histogram.h"

#include <TFCommon.h>
#include <TFWorkCopy.h>

#include <QtGui/QPainter>


namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFAbstractPainter{

public:

	typedef typename boost::shared_ptr<TFAbstractPainter<dim>> Ptr;

	virtual void setArea(QRect area) = 0;
	virtual QRect getInputArea() = 0;

	virtual QPixmap getView(typename TFWorkCopy<dim>::Ptr workCopy) = 0;

protected:

	QRect area_;

	TFAbstractPainter(){}
	virtual ~TFAbstractPainter(){};
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER