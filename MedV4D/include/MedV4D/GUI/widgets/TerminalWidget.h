#ifndef TERMINAL_WIDGET_HPP
#define TERMINAL_WIDGET_HPP

#include "MedV4D/Common/Common.h"
#include <QtWidgets>
#include <list>


namespace M4D
{
namespace GUI
{

class PromptLineWidget: public QTextEdit
{
	Q_OBJECT;
public:
	PromptLineWidget( QWidget *aParent = NULL ): QTextEdit( aParent )
	{
		setMinimumHeight( 25 );
		setVerticalScrollBarPolicy( Qt::ScrollBarAlwaysOn );
		setSizePolicy( QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Ignored ) );
		//QObject::connect( this, SIGNAL( textChanged() ), this, SLOT( resizeToText() ) );
	}
public slots:
	void
	resizeToText()
	{
		//TODO
		LOG( "height " << contentsRect().height() );
		LOG( "width " << contentsRect().width() );

		int diff = verticalScrollBar()->maximum() - verticalScrollBar()->minimum();
		if( diff ) {
			setMinimumHeight( diff * 3 );
		} else {
			setMinimumHeight( 25 );
		}
	}
signals:
	void
	inputConfirmed( const QString &str );

	void
	getPreviousCommand();

	void
	getNextCommand();
protected:
	void
	keyPressEvent ( QKeyEvent * e )
	{
		switch( e->key() ) {
		case Qt::Key_Enter:
		case Qt::Key_Return:
			if ( e->modifiers() & Qt::ShiftModifier ) {
				QTextEdit::keyPressEvent( e );
			} else {
				emit inputConfirmed( toPlainText() );
			}
			break;
		case Qt::Key_Up:
			emit getPreviousCommand();
			break;
		case Qt::Key_Down:
			emit getNextCommand();
			break;
		default:
			QTextEdit::keyPressEvent( e );
			break;
		}
	}

	/*void
	keyReleaseEvent ( QKeyEvent * e )
	{

	}*/
};


class TerminalWidget: public QWidget
{
	Q_OBJECT;
public:
	TerminalWidget( QWidget *aParent = NULL ): QWidget( aParent )
	{
		mPromptText = ">>>";

		QVBoxLayout *layout1 = new QVBoxLayout;
		QHBoxLayout *layout2 = new QHBoxLayout;

		mOutput = new QTextEdit;
		mOutput->setReadOnly( true );
		mOutput->setLineWrapMode( QTextEdit::WidgetWidth );

		mPromptLabel = new QLabel( mPromptText );
		mPromptLabel->setAlignment( Qt::AlignTop | Qt::AlignHCenter );
		mPromptLine = new PromptLineWidget;
		//mPromptLine->setminimumSizeHint( QSize( 300, 10 ) );
		mOutput->setSizePolicy( QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum ) );

		layout1->addWidget( mOutput, 10 );
		layout1->setSpacing( 0 );
		layout1->addLayout( layout2, 0 );


		layout2->addWidget( mPromptLabel );
		layout2->addWidget( mPromptLine );

		setLayout( layout1 );

		QObject::connect( mPromptLine, SIGNAL( inputConfirmed( const QString & ) ), this, SLOT( inputConfirmed( const QString & ) ) );
		QObject::connect( mPromptLine, SIGNAL( getPreviousCommand() ), this, SLOT( getPreviousCommand() ) );
		QObject::connect( mPromptLine, SIGNAL( getNextCommand() ), this, SLOT( getNextCommand() ) );

		QFont font("Monospace");
		font.setStyleHint(QFont::TypeWriter);
		mPromptLine->setFont( font );
		mOutput->setFont( font );
		mOutput->setTabStopWidth( 40 );
		mPromptLine->setTabStopWidth( 40 );


		mLastCommand = mCommands.end();

		mDefaultCharFormat = mOutput->currentCharFormat();
		mErrorCharFormat = mOutput->currentCharFormat();
		mErrorCharFormat.setForeground( QColor( 255, 0, 0, 255 ) );
	}

	~TerminalWidget()
	{
	}

protected slots:

	void
	inputConfirmed( const QString &str )
	{
		mOutput->moveCursor( QTextCursor::End, QTextCursor::MoveAnchor );
		mOutput->textCursor().insertText( mPromptText  + str + "\n", mDefaultCharFormat );
		mOutput->ensureCursorVisible();

		if ( !str.isEmpty() && //TODO check for whitespace only strings
			( mCommands.empty() || str != mCommands.back() ) )
		{
			mCommands.push_back( str );
		}
		mLastCommand = mCommands.end();
		mPromptLine->clear();
		processInput( str );
	}

	void
	appendText( const QString &str )
	{
		mOutput->moveCursor( QTextCursor::End, QTextCursor::MoveAnchor );
		mOutput->textCursor().insertText( str, mDefaultCharFormat );
		mOutput->ensureCursorVisible();
		//mOutput->insertPlainText( str );
	}

	void
	appendErrorText( const QString &str )
	{
		mOutput->moveCursor( QTextCursor::End, QTextCursor::MoveAnchor );
		mOutput->textCursor().insertText( str, mErrorCharFormat );
		mOutput->ensureCursorVisible();
		//appendText( str );
	}

	void
	getPreviousCommand()
	{
		if( mCommands.empty() ) {
			return;
		}
		if( mLastCommand != mCommands.begin() ) {
			--mLastCommand;
		}
		if( mLastCommand != mCommands.end() ) {
			mPromptLine->clear();
			mPromptLine->setPlainText( *mLastCommand );
			mPromptLine->moveCursor( QTextCursor::End, QTextCursor::MoveAnchor );
		}

	}

	void
	getNextCommand()
	{
		if( mCommands.empty() ) {
			return;
		}
		if( mLastCommand != mCommands.end() ) {
			++mLastCommand;
		}
		mPromptLine->clear();
		if( mLastCommand != mCommands.end() ) {
			mPromptLine->setPlainText( *mLastCommand );
			mPromptLine->moveCursor( QTextCursor::End, QTextCursor::MoveAnchor );
		}


	}
protected:
	virtual void
	processInput( const QString &str ) = 0;



	QString mPromptText;

	QTextEdit *mOutput;
	PromptLineWidget *mPromptLine;
	QLabel *mPromptLabel;

	typedef std::list< QString > CommandList;
	CommandList mCommands;
	//QStringList mCommands;
	CommandList::iterator mLastCommand;

	QTextCharFormat mErrorCharFormat;
	QTextCharFormat mDefaultCharFormat;
};

}//GUI
}//M4D


#endif /*TERMINAL_WIDGET_HPP*/
