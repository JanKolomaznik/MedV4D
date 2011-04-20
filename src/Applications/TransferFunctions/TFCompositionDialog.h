#ifndef TF_COMPOSITION_DIALOG
#define TF_COMPOSITION_DIALOG

#include <set>

#include <QtGui/QDialog>
#include <QtGui/QVBoxLayout>
#include <QtGui/QCheckBox>

#include <TFCommon.h>
#include <TFPaletteButton.h>

#include <ui_TFCompositionDialog.h>

namespace M4D {
namespace GUI {

class TFPalette;
class TFBasicHolder;

class TFCompositionDialog: public QDialog{

	Q_OBJECT

public:

	typedef std::set<TF::Size> Selection;

	TFCompositionDialog(QWidget* parent = 0);
	~TFCompositionDialog();

	void updateSelection(const std::map<TF::Size, TFBasicHolder*>& editors, TFPalette* palette);
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

	typedef std::map<TF::Size, TFPaletteCheckButton*> Buttons;

	bool previewEnabled_;

	TF::Size colModulator_;

	Ui::TFCompositionDialog* ui_;
	QGridLayout* layout_;
	QSpacerItem* pushUpSpacer_;

	bool selectionChanged_;

	Selection indexesMemory_;
	Selection indexes_;
	
	Buttons buttons_;

	void clearLayout_();
};

} // namespace GUI
} // namespace M4D

#endif	//TF_COMPOSITION_DIALOG