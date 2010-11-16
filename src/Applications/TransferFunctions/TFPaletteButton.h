#ifndef TF_PALETTE_BUTTON
#define TF_PALETTE_BUTTON

#include <TFTypes.h>
#include <TFAbstractHolder.h>
#include <QtGui/QPushButton>

namespace M4D {
namespace GUI {

class TFPaletteButton: public QPushButton{

	Q_OBJECT

public:
	TFPaletteButton(QWidget* parent, TFAbstractHolder* holder, TFSize index);

	~TFPaletteButton();

	void setUpHolder(QWidget* parent);
	void hideHolder();
	void showHolder();

	void saveHolder();

	void changeIndex(TFSize index);

	template<typename ElementIterator>
	bool applyTransferFunction(
		ElementIterator begin,
		ElementIterator end){

		if(!holder_) 
		{
			QMessageBox::critical(this, QObject::tr("Transfer Functions"),
				QObject::tr("No function available!"));
			return false;
		}
		return holder_->applyTransferFunction<ElementIterator>(begin, end);
	}

signals:
	void TFPaletteSignal(const TFSize &index);

public slots:
	void button_triggered();

private:	
	TFSize index_;

	TFAbstractHolder* holder_;
};

typedef std::vector<TFPaletteButton*> TFPalette;
typedef std::vector<TFPaletteButton*>::iterator TFPaletteIt;

} // namespace GUI
} // namespace M4D

#endif	//TF_PALETTE_BUTTON