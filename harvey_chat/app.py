"""Terminal UI for chatting with Harvey-LLM."""

from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from harvey_chat.config import BANNER, WELCOME
from harvey_chat.model import HarveyModel, check_platform


class ChatMessage(Static):
    """Single chat bubble."""

    def __init__(self, speaker: str, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.speaker = speaker
        self.text = text

    def render(self) -> str:
        color = "bold cyan" if self.speaker == "HARVEY" else "bold white"
        return f"[{color}]{self.speaker}:[/] {self.text}"


class HarveyApp(App):
    """Harvey Specter chat TUI."""

    CSS = """
    Screen {
        background: #0a0a0a;
    }
    #banner {
        color: #c9a227;
        text-align: center;
        padding: 1 0;
        background: #111;
        border-bottom: solid #333;
    }
    #chat {
        height: 1fr;
        padding: 1 2;
    }
    #input-box {
        dock: bottom;
        margin: 0 1 1 1;
        border: tall #c9a227;
        background: #111;
    }
    ChatMessage {
        margin-bottom: 1;
        padding: 0 1;
        width: 100%;
    }
    #status {
        dock: bottom;
        color: #888;
        padding: 0 2;
        background: #111;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.harvey = HarveyModel()
        self._generating = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(BANNER, id="banner")
        yield VerticalScroll(id="chat")
        yield Static("Initializing…", id="status")
        yield Input(placeholder="Say something to Harvey…", id="input-box")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input-box", Input).disabled = True
        for warning in check_platform():
            self._add_message("SYSTEM", warning)
        self._add_message("SYSTEM", WELCOME)
        self.load_model()

    def _add_message(self, speaker: str, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(ChatMessage(speaker, text))
        chat.scroll_end(animate=False)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    @work(thread=True)
    def load_model(self) -> None:
        try:
            msg = self.harvey.load()
            self.call_from_thread(self._on_model_ready, msg)
        except Exception as exc:
            self.call_from_thread(self._on_model_error, str(exc))

    def _on_model_ready(self, msg: str) -> None:
        self._set_status(msg)
        self.query_one("#input-box", Input).disabled = False
        self.query_one("#input-box", Input).focus()

    def _on_model_error(self, error: str) -> None:
        self._set_status(f"Load failed: {error}")
        self._add_message("SYSTEM", f"Error loading model: {error}")

    @on(Input.Submitted, "#input-box")
    def handle_input(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self._generating:
            return

        if text.lower() in {"/quit", "/exit", "/q"}:
            self.exit()
            return
        if text.lower() == "/clear":
            self.harvey.clear_history()
            self.query_one("#chat", VerticalScroll).clear()
            self._add_message("SYSTEM", "Conversation cleared.")
            return

        self._generating = True
        self.query_one("#input-box", Input).disabled = True
        self._add_message("YOU", text)
        self.generate_reply(text)

    @work(thread=True)
    def generate_reply(self, user_text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        reply_widget = ChatMessage("HARVEY", "")
        self.call_from_thread(chat.mount, reply_widget)
        self.call_from_thread(chat.scroll_end, animate=False)

        try:
            tokens: list[str] = []
            for token in self.harvey.stream_reply(user_text):
                tokens.append(token)
                full = "".join(tokens)
                self.call_from_thread(
                    reply_widget.update,
                    f"[bold cyan]HARVEY:[/] {full}",
                )
                self.call_from_thread(chat.scroll_end, animate=False)
        except Exception as exc:
            self.call_from_thread(
                self._add_message, "SYSTEM", f"Generation error: {exc}"
            )
        finally:
            self.call_from_thread(self._finish_generation)

    def _finish_generation(self) -> None:
        self._generating = False
        self.query_one("#input-box", Input).disabled = False
        self.query_one("#input-box", Input).focus()
        self._set_status("Ready.")

    def action_clear_chat(self) -> None:
        self.harvey.clear_history()
        self.query_one("#chat", VerticalScroll).clear()
        self._add_message("SYSTEM", "Conversation cleared.")


def run() -> None:
    HarveyApp().run()
