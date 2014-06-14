#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include <QtCore/QString>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/WorkCopy.h"

#include <QPainter>


namespace M4D {
namespace GUI {

class AbstractPainter
{
public:
	typedef std::shared_ptr<AbstractPainter> Ptr;

	virtual void setArea(QRect area) = 0;
	virtual QRect getInputArea() = 0;

	virtual QPixmap getView(WorkCopy::Ptr workCopy) = 0;

	virtual void save(TF::XmlWriterInterface* writer){}
	virtual bool load(TF::XmlReaderInterface* reader){

		#ifndef TF_NDEBUG
			std::cout << "Loading painter..." << std::endl;
		#endif

		return true;
	}

protected:

	QRect area_;

	AbstractPainter(){}
	virtual ~AbstractPainter(){};
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER
