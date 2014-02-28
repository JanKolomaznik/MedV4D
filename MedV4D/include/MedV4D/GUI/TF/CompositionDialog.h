#ifndef TF_COMPOSITION_DIALOG
#define TF_COMPOSITION_DIALOG

#include <set>

#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QCheckBox>

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/PaletteButton.h"

#include "MedV4D/generated/ui_CompositionDialog.h"

namespace M4D {
namespace GUI {

class Palette;
class Editor;

class CompositionDialog: public QDialog{

	Q_OBJECT

public:

	typedef std::set<TF::Size> Selection;

	CompositionDialog(QWidget* parent = 0);
	~CompositionDialog();

	void updateSelection(const std::map<TF::Size, Editor*>& editors, Palette* palette);
	bool selectionChanged();
	Selection getComposition();

	virtual void accept();
	virtual void reject();

private slots:

	void on_previewsCheck_toggled(bool enable);
	void button_triggered(TF::Size index);

protected:

	void resizeEvent(QResizeEvent*);

private:

	typedef std::map<TF::Size, PaletteCheckButton*> Buttons;

	bool previewEnabled_;

	Ui::CompositionDialog* ui_;
	QGridLayout* layout_;
	TF::Size colModulator_;

	bool selectionChanged_;

	Selection indexesMemory_;
	Selection indexes_;

	Buttons buttons_;

	void clearLayout_();
};

} // namespace GUI
} // namespace M4D

#endif	//TF_COMPOSITION_DIALOG
