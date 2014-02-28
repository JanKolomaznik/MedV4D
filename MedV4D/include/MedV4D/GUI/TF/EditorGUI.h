#ifndef TF_BASICHOLDER
#define TF_BASICHOLDER

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QMainWindow>

#include <QtWidgets/QMessageBox>

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/Editor.h"

#include "MedV4D/generated/ui_EditorGUI.h"

namespace M4D {
namespace GUI {

class EditorGUI: public Editor{

	Q_OBJECT

public:

	EditorGUI(AbstractModifier::Ptr modifier,
		TF::Types::Structure structure,
		Attributes attributes,
		std::string name);

	~EditorGUI();

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

	Ui::EditorGUI* ui_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_BASICHOLDER
