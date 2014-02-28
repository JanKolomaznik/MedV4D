#ifndef TF_ABSTRACT_MODIFIER
#define TF_ABSTRACT_MODIFIER

#include <QtCore/QString>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/WorkCopy.h"
#include "MedV4D/GUI/TF/AbstractPainter.h"
#include "MedV4D/GUI/TF/AbstractFunction.h"

#include <QtWidgets/QWidget>

namespace M4D {
namespace GUI {

class AbstractModifier: public QWidget{

public:

	typedef boost::shared_ptr<AbstractModifier> Ptr;

	virtual
	~AbstractModifier(){}

	virtual void
	setDataStructure(const std::vector<TF::Size>& dataStructure) = 0;

	void
	setHistogram(const TF::HistogramInterface::Ptr histogram);

	QWidget*
	getTools();

	TF::Size
	getDimension();

	FunctionInterface::Const
	getFunction();

	bool
	changed();

	M4D::Common::TimeStamp
	getTimeStamp();

	virtual void
	save(TF::XmlWriterInterface* writer);

	void
	saveFunction(TF::XmlWriterInterface* writer);

	bool
	load(TF::XmlReaderInterface* reader, bool& sideError);

	bool
	loadFunction(TF::XmlReaderInterface* reader);

protected:

	QWidget* toolsWidget_;

	AbstractPainter::Ptr painter_;

	bool changed_;
	M4D::Common::TimeStamp stamp_;

	WorkCopy::Ptr workCopy_;
	TF::Coordinates coords_;

	QRect inputArea_;
	const TF::PaintingPoint ignorePoint_;

	AbstractModifier(FunctionInterface::Ptr function, AbstractPainter::Ptr painter);

	virtual void createTools_() = 0;

	virtual void computeInput_() = 0;

	void resizeEvent(QResizeEvent* e);
	void paintEvent(QPaintEvent*);

	virtual void mousePressEvent(QMouseEvent *e){}
	virtual void mouseReleaseEvent(QMouseEvent *e){}
	virtual void mouseMoveEvent(QMouseEvent *e){}
	virtual void wheelEvent(QWheelEvent *e){}

	virtual void keyPressEvent(QKeyEvent *e){}
	virtual void keyReleaseEvent(QKeyEvent *e){}

	virtual void saveSettings_(TF::XmlWriterInterface* writer){}
	virtual bool loadSettings_(TF::XmlReaderInterface* reader){ return true; }
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_MODIFIER
