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

#include <QtGui/QWidget>

namespace M4D {
namespace GUI {

class TFAbstractModifier: public QWidget{

public:

	typedef boost::shared_ptr<TFAbstractModifier> Ptr;

	virtual ~TFAbstractModifier(){}

	virtual void setDataStructure(const std::vector<TF::Size>& dataStructure) = 0;
	virtual void setHistogram(const TF::Histogram::Ptr histogram) = 0;

	QWidget* getTools(){

		if(!toolsWidget_) createTools_();
		return toolsWidget_;
	}

	TF::Size getDimension(){

		return workCopy_->getDimension();
	}

	TFFunctionInterface::Const getFunction(){

		return workCopy_->getFunction();
	}

	bool changed(){

		if(changed_)
		{
			changed_ = false;
			return true;
		}
		return false;
	}

	M4D::Common::TimeStamp getTimeStamp(){

		return stamp_;
	}

	virtual void save(TF::XmlWriterInterface* writer){

		saveSettings_(writer);
		painter_->save(writer);
		workCopy_->save(writer);
	}

	bool load(TF::XmlReaderInterface* reader, bool& sideError){

		#ifndef TF_NDEBUG
			std::cout << "Loading modifier..." << std::endl;
		#endif
	
		sideError = loadSettings_(reader);

		bool error = painter_->load(reader);
		sideError = sideError || error;

		bool ok = workCopy_->load(reader, error);
		sideError = sideError || error;

		return ok;
	}

protected:

	QWidget* toolsWidget_;

	TFAbstractPainter::Ptr painter_;

	bool changed_;
	M4D::Common::TimeStamp stamp_;

	TFWorkCopy::Ptr workCopy_;
	QRect inputArea_;
	const TF::PaintingPoint ignorePoint_;

	TFAbstractModifier(TFFunctionInterface::Ptr function, TFAbstractPainter::Ptr painter):
		painter_(painter),
		workCopy_(TFWorkCopy::Ptr(new TFWorkCopy(function))),
		ignorePoint_(-1, -1),
		toolsWidget_(NULL),
		changed_(true){
	}

	virtual void createTools_() = 0;

	void resizeEvent(QResizeEvent* e){

		painter_->setArea(rect());

		inputArea_ = painter_->getInputArea();

		computeInput_();
		update();
	}

	void paintEvent(QPaintEvent*){

		QPainter drawer(this);
		drawer.drawPixmap(rect(), painter_->getView(workCopy_));
	}

	virtual void computeInput_() = 0;

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