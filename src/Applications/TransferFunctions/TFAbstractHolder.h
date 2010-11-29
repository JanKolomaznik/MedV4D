#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "ui_TFAbstractHolder.h"

#include <QtGui/QWidget>
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

class TFAbstractHolder : public QWidget{

	Q_OBJECT
	
	friend class TFHolderFactory;

public:

	virtual ~TFAbstractHolder();

	virtual void save();

	virtual void setUp(const TFSize& index) = 0;
	bool connectToTFWindow(QObject* tfWindow);	/*	tfWindow has to be TFWindow instance,
													need to call method createButton in order to connect palette button	*/
	void createPaletteButton(QWidget* parent);
	void changeIndex(const TFSize& index);

	TFHolderType getType() const;
	TFPaletteButton* getButton() const;
	bool isReleased();

	void setReleased(const bool& released);
	
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

	void CloseHolder();
	void ReleaseHolder();
	void ActivateHolder(const TFSize& index);

protected slots:

	void size_changed(const TFSize& index, const QRect& rect);
	void on_releaseButton_clicked();
	void on_closeButton_clicked();

protected:

	TFHolderType type_;
	Ui::TFAbstractHolder* basicTools_;
	bool setup_;
	TFPaintingPoint painterLeftTop_;
	TFPaintingPoint painterRightBottom_;
	TFSize colorBarSize_;
	TFPaletteButton* button_;
	TFSize index_;
	bool released_;

	void mousePressEvent(QMouseEvent*);
	void keyPressEvent(QKeyEvent*);
	virtual void paintEvent(QPaintEvent*);
	virtual void resizeEvent(QResizeEvent*);

	virtual bool load_(QFile &file);
	virtual void save_(QFile &file);

	virtual void updateFunction_() = 0;
	virtual void updatePainter_() = 0;
	virtual void resizePainter_(const QRect& rect) = 0;

	virtual TFAbstractFunction* getFunction_() = 0;

	void calculate_(const TFColorMapPtr input, TFColorMapPtr output);

	TFAbstractHolder();
	TFAbstractHolder(QWidget* widget);
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTHOLDER