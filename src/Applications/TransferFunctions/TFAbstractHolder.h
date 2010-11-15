#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "ui_TFAbstractHolder.h"

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <QtCore/QString>

#include <TFTypes.h>
#include <TFXmlWriter.h>
#include <TFXmlReader.h>

#include <TFHSVaFunction.h>
#include <TFRGBaFunction.h>

namespace M4D {
namespace GUI {

class TFAbstractHolder : public QWidget{

	Q_OBJECT
	
	friend class TFHolderFactory;

public:

	virtual ~TFAbstractHolder();

	virtual void save();

	virtual void setUp(QWidget *parent, const QRect rect) = 0;

	TFHolderType getType() const;
	
	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		TFAbstractFunction* transferFunction = getFunction_();
		updateFunction_();
		switch(transferFunction->getType()){
			case TFFUNCTION_RGBA:
			{
				TFRGBaFunction* rgbaFunction = dynamic_cast<TFRGBaFunction*>(transferFunction);
				return rgbaFunction->apply<ElementIterator>(begin, end);
			}
			case TFFUNCTION_HSVA:
			{
				TFHSVaFunction* hsvaFunction = dynamic_cast<TFHSVaFunction*>(transferFunction);
				return hsvaFunction->apply<ElementIterator>(begin, end);
			}
			case TFFUNCTION_UNKNOWN:
			default:
			{
				tfAssert("Unknown Transfer Function");
				break;
			}
		}
		return false;
	}

protected slots:
	void size_changed(const QRect rect);

protected:
	TFHolderType type_;
	Ui::TFAbstractHolder* basicTools_;
	bool setup_;
	TFPaintingPoint painterLeftTop_;
	TFPaintingPoint painterRightBottom_;

	virtual void paintEvent(QPaintEvent *e);

	virtual bool load_(QFile &file);
	virtual void save_(QFile &file);

	virtual void updateFunction_() = 0;
	virtual void updatePainter_() = 0;
	virtual void resizePainter_(const QRect& rect) = 0;

	virtual TFAbstractFunction* getFunction_() = 0;

	void calculate_(const TFColorMapPtr input, TFColorMapPtr output);

	TFAbstractHolder();
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTHOLDER