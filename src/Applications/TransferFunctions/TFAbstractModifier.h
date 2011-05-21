#ifndef TF_ABSTRACT_MODIFIER
#define TF_ABSTRACT_MODIFIER

#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>

#include <TFCommon.h>
#include <TFWorkCopy.h>
#include <TFAbstractPainter.h>
#include <TFAbstractFunction.h>

#include <QtGui/QWidget>

namespace M4D {
namespace GUI {

class TFAbstractModifier: public QWidget{

public:

	typedef boost::shared_ptr<TFAbstractModifier> Ptr;

	virtual ~TFAbstractModifier(){}

	virtual void setDataStructure(const std::vector<TF::Size>& dataStructure) = 0;

	void setHistogram(const TF::HistogramInterface::Ptr histogram);

	QWidget* getTools();
	TF::Size getDimension();
	TFFunctionInterface::Const getFunction();

	bool changed();
	M4D::Common::TimeStamp getTimeStamp();

	virtual void save(TF::XmlWriterInterface* writer);
	void saveFunction(TF::XmlWriterInterface* writer);
	bool load(TF::XmlReaderInterface* reader, bool& sideError);
	bool loadFunction(TF::XmlReaderInterface* reader);

protected:

	QWidget* toolsWidget_;

	TFAbstractPainter::Ptr painter_;

	bool changed_;
	M4D::Common::TimeStamp stamp_;

	TFWorkCopy::Ptr workCopy_;
	TF::Coordinates coords_;

	QRect inputArea_;
	const TF::PaintingPoint ignorePoint_;

	TFAbstractModifier(TFFunctionInterface::Ptr function, TFAbstractPainter::Ptr painter);

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