#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include <TFXmlReader.h>
#include <TFXmlWriter.h>
#include <QtCore/QString>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>

#include <TFCommon.h>
#include <TFWorkCopy.h>

#include <QtGui/QPainter>


namespace M4D {
namespace GUI {

class TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFAbstractPainter> Ptr;

	virtual void setArea(QRect area) = 0;
	virtual QRect getInputArea() = 0;

	virtual QPixmap getView(TFWorkCopy::Ptr workCopy) = 0;

	virtual void save(TFXmlWriter::Ptr writer){}
	virtual bool load(TFXmlReader::Ptr reader){

		#ifndef TF_NDEBUG
			std::cout << "Loading painter..." << std::endl;
		#endif

		return true;
	}

protected:

	QRect area_;

	TFAbstractPainter(){}
	virtual ~TFAbstractPainter(){};
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER