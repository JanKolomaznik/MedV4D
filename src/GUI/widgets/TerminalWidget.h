#ifndef TERMINAL_WIDGET_HPP
#define TERMINAL_WIDGET_HPP

#include <QtGui>

namespace M4D
{
namespace GUI
{

class TerminalWidget: public QWidget
{
	Q_OBJECT;
public:
	TerminalWidget( QWidget *aParent = NULL ): QWidget( aParent )
	{
		QVBoxLayout *layout1 = new QVBoxLayout;
		QHBoxLayout *layout2 = new QHBoxLayout;

		mOutput = new QTextEdit;
		mOutput->setReadOnly( true );
		mOutput->setLineWrapMode( QTextEdit::WidgetWidth );

		mPromptLabel = new QLabel( ">>>" );
		mPromptLine = new QTextEdit;
		mPromptLine->setMinimumHeight( 10 );
		//mPromptLine->setminimumSizeHint( QSize( 300, 10 ) );
		mPromptLine->setSizePolicy( QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred ) );

		layout1->addWidget( mOutput, 10 );
		layout1->setSpacing( 0 );
		layout1->addLayout( layout2, 0 );


		layout2->addWidget( mPromptLabel );
		layout2->addWidget( mPromptLine );

		setLayout( layout1 );
	}
protected:

	QTextEdit *mOutput;
	QTextEdit *mPromptLine;
	QLabel *mPromptLabel;
};

}//GUI
}//M4D


#endif /*TERMINAL_WIDGET_HPP*/
