#ifndef TF_COMPOSITION_DIALOG
#define TF_COMPOSITION_DIALOG

#include <set>

#include <QtGui/QDialog>
#include <QtGui/QVBoxLayout>
#include <QtGui/QCheckBox>

#include <TFCommon.h>

#include <ui_TFCompositionDialog.h>

namespace M4D {
namespace GUI {

class TFPalette;
class TFBasicHolder;

class TFCompositionDialog: public QDialog{

	Q_OBJECT

public:

	typedef std::vector<TFBasicHolder*> Composition;

	TFCompositionDialog(TFPalette* palette);
	~TFCompositionDialog();

	bool refreshSelection();
	Composition getComposition();

//private slots:

	//void on_toggleView_clicked();	//zapnuti/vypnuti nahledu

private:	

	Ui::TFCompositionDialog* ui_;
	QVBoxLayout* layout_;
	QSpacerItem* pushUpSpacer_;

	std::vector<QCheckBox*> checkBoxes_;
	Composition allAvailableEditors_;
	std::set<TF::Size> indexesMemory_;

	TFPalette* palette_;

	Common::TimeStamp lastPaletteChange_;

	void clearLayout_();
};

} // namespace GUI
} // namespace M4D

#endif	//TF_COMPOSITION_DIALOG