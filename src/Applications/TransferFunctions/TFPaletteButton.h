#ifndef TF_PALETTE_BUTTON
#define TF_PALETTE_BUTTON

#include <TFTypes.h>
#include <QtGui/QPushButton>

namespace M4D {
namespace GUI {

class TFPaletteButton: public QPushButton{

	Q_OBJECT

public:

	TFPaletteButton(QWidget* parent, TFSize index);

	~TFPaletteButton();

	void changeIndex(TFSize index);

signals:

	void TFPaletteSignal(const TFSize &index);

public slots:

	void button_triggered();

private:

	TFSize index_;
};

} // namespace GUI
} // namespace M4D

#endif	//TF_PALETTE_BUTTON