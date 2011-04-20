#ifndef TF_COMPOSITE_MODIFIER
#define TF_COMPOSITE_MODIFIER

#include <TFViewModifier.h>
#include <TFCompositionDialog.h>

#include <QtCore/QTimer>
#include <QtGui/QVBoxLayout>

#include <ui_TFCompositeModifier.h>

namespace M4D {
namespace GUI {

class TFPalette;

class TFCompositeModifier: public TFViewModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFCompositeModifier> Ptr;

	TFCompositeModifier(
		TFAbstractFunction<TF_DIMENSION_1>::Ptr function,
		TFSimplePainter::Ptr painter,		
		TFPalette* palette);

	~TFCompositeModifier();

protected slots:

	void manageComposition_clicked();
	void change_check();
	void changeChecker_intervalChange(int value);

protected:

	struct Editor{
		TFBasicHolder* holder;
		Common::TimeStamp change;
		QLabel* name;

		void updateName();

		Editor(TFBasicHolder* holder);

		~Editor();
	};
	typedef std::map<TF::Size, Editor*> Composition;

	typedef TFCompositionDialog::Selection Selection;

	Ui::TFCompositeModifier* compositeTools_;
	QWidget* compositeWidget_;
	QVBoxLayout* layout_;
	QSpacerItem* pushUpSpacer_;
	
	TFPalette* palette_;
	Common::TimeStamp lastPaletteChange_;
	std::map<TF::Size, TFBasicHolder*> editors_;

	TFCompositionDialog manager_;
	bool managing_;

	QTimer changeChecker_;
	Composition composition_;

	TFAbstractFunction<TF_DIMENSION_1>::Ptr function_;

	virtual void computeResultFunction_();
	void updateComposition_();

	virtual void createTools_();
	void clearLayout_();

	void computeInput_();
	std::vector<int> computeZoomMoveIncrements_(const int moveX, const int moveY);

	virtual void wheelEvent(QWheelEvent* e);
};

} // namespace GUI
} // namespace M4D

#endif //TF_COMPOSITE_MODIFIER