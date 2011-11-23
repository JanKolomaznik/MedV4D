#ifndef TF_BASICHOLDER
#define TF_BASICHOLDER

#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>

#include <QtGui/QMessageBox>

#include <TFCommon.h>
#include <TFEditor.h>

#include <ui_TFEditorGUI.h>

namespace M4D {
namespace GUI {	

class TFEditorGUI: public TFEditor{

	Q_OBJECT

public:
	
	TFEditorGUI(TFAbstractModifier::Ptr modifier,
		TF::Types::Structure structure,
		Attributes attributes,
		std::string name);

	~TFEditorGUI();

	void setup(QMainWindow* mainWindow, const int index = -1);

	void setActive(const bool active);
	void setAvailable(const bool available);

protected slots:

	void on_actionEditorSave_triggered();
	void on_actionEditorSaveAs_triggered();
	void on_actionFunctionSave_triggered();
	void on_actionFunctionSaveAs_triggered();
	void on_actionFunctionLoad_triggered();
	void on_actionClose_triggered();

	void on_activateButton_clicked();

	void on_nameEdit_editingFinished();

protected:

	Ui::TFEditorGUI* ui_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_BASICHOLDER
