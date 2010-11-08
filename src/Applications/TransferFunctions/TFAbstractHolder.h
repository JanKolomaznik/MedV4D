#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "ui_TFAbstractHolder.h"

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <QtCore/QString>

#include <TFTypes.h>
#include <TFSimpleFunction.h>
#include <TFGrayscaleTransparencyFunction.h>
#include <TFRGBFunction.h>
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

	TFType getType() const;
	
	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		TFAbstractFunction* transferFunction = getFunction_();
		updateFunction_();
		switch(transferFunction->getType()){
			case TFTYPE_SIMPLE:
			{
				TFSimpleFunction* simpleFunction = dynamic_cast<TFSimpleFunction*>(transferFunction);
				return simpleFunction->apply<ElementIterator>(begin, end);
			}
			case TFTYPE_GRAYSCALE_TRANSPARENCY:
			{
				TFGrayscaleTransparencyFunction* grayTransFunction = dynamic_cast<TFGrayscaleTransparencyFunction*>(transferFunction);
				return grayTransFunction->apply<ElementIterator>(begin, end);
			}
			case TFTYPE_RGB:
			{
				TFRGBFunction* rgbFunction = dynamic_cast<TFRGBFunction*>(transferFunction);
				return rgbFunction->apply<ElementIterator>(begin, end);
			}
			case TFTYPE_RGBA:
			{
				TFRGBaFunction* rgbaFunction = dynamic_cast<TFRGBaFunction*>(transferFunction);
				return rgbaFunction->apply<ElementIterator>(begin, end);
			}
			case TFTYPE_UNKNOWN:
			default:
			{
				tfAssert("Unknown Transfer Function");
				break;
			}
		}
		return false;
	}

	//virtual void receiveHistogram(const TFHistogram& histogram){};

protected slots:
	virtual void size_changed(const QRect rect) = 0;

protected:
	TFType type_;
	Ui::TFAbstractHolder* basicTools_;
	bool setup_;

	virtual bool load_(QFile &file) = 0;
	virtual void save_(QFile &file) = 0;

	virtual void updateFunction_() = 0;

	virtual TFAbstractFunction* getFunction_() = 0;

	void calculate_(const TFFunctionMapPtr input, TFFunctionMapPtr output);

	TFAbstractHolder();
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTHOLDER