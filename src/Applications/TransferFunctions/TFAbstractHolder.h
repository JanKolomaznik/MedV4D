#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "ui_TFAbstractHolder.h"

#include <QtGui/QDockWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QPainter>

#include <QtCore/QString>

#include <TFTypes.h>
#include <TFXmlWriter.h>
#include <TFXmlReader.h>
#include <TFPaletteButton.h>

#include <TFHSVaFunction.h>
#include <TFRGBaFunction.h>

namespace M4D {
namespace GUI {

class TFAbstractHolder : public QDockWidget{

	Q_OBJECT
	
	friend class TFHolderFactory;

public:

	virtual ~TFAbstractHolder();

	void save();

	/*virtual*/ void setUp(TFSize index);
	bool connectToTFPalette(QObject* tfPalette);	//	tfPalette has to be TFPalette instance
	bool createPaletteButton(QWidget* parent);
	//void changeIndex(TFSize index);
	TFSize getIndex();

	TFHolderType getType() const;
	TFPaletteButton* getButton() const;
	
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

signals:

	void Close(TFSize index);
	void Activate(TFSize index);

protected slots:

	void on_closeButton_clicked();
	void on_saveButton_clicked();
	void on_activateButton_clicked();

protected:

	TFHolderType type_;
	Ui::TFAbstractHolder* basicTools_;
	bool setup_;
	const int bottomSpace_;
	TFPaletteButton* button_;
	TFSize index_;

	virtual void paintEvent(QPaintEvent*);
	virtual void resizeEvent(QResizeEvent*);

	virtual bool load_(QFile &file);
	virtual void save_(QFile &file);

	virtual void updateFunction_() = 0;
	virtual void updatePainter_() = 0;
	virtual void resizePainter_() = 0;

	virtual TFAbstractFunction* getFunction_() = 0;

	void calculate_(const TFColorMapPtr input, TFColorMapPtr output);

	TFAbstractHolder();
	TFAbstractHolder(QMainWindow* widget);
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTHOLDER