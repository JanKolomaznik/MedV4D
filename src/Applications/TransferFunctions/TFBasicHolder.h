#ifndef TF_BASICHOLDER
#define TF_BASICHOLDER

#include "common/Types.h"

#include <QtGui/QMainWindow>
#include <QtGui/QDockWidget>

#include <QtGui/QPainter>

#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtGui/QPaintEvent>

#include <QtCore/QString>

#include <TFCommon.h>
#include <TFAbstractHolder.h>

namespace M4D {
namespace GUI {

class TFBasicHolder: public TFAbstractHolder{

public:

	typedef boost::shared_ptr<TFBasicHolder> Ptr;

	TFBasicHolder(TFAbstractPainter<1>::Ptr painter,
		TFAbstractModifier<1>::Ptr modifier,
		TF::Types::Structure structure);

	~TFBasicHolder();

	bool loadData(TFXmlReader::Ptr reader, bool& sideError);

	void setup(QMainWindow* mainWindow, const int index = -1);
	void setHistogram(TF::Histogram::Ptr histogram);
	void setDomain(const TF::Size domain);

	bool changed();

protected:

	QDockWidget* dockTools_;

	TFAbstractModifier<1>::Ptr modifier_;
	TFAbstractPainter<1>::Ptr painter_;

	TFApplyFunctionInterface::Ptr functionToApply_();

	void saveData_(TFXmlWriter::Ptr writer);

	void paintEvent(QPaintEvent*);
	void resizeEvent(QResizeEvent*);
};

} // namespace GUI
} // namespace M4D

#endif //TF_BASICHOLDER
