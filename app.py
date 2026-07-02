import sys
import subprocess
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Button, Static, Label
from textual.screen import Screen

class MainMenu(Container):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("FaceFlow - Deepfake Data Harvester", id="title"),
            Label("Select an operation to perform:", id="subtitle"),
            Button("1. Run YouTube Shorts Downloader", id="btn_yt", variant="primary"),
            Button("2. Run TikTok Hashtag Downloader", id="btn_tk1", variant="primary"),
            Button("3. Run TikTok Account Downloader", id="btn_tk2", variant="primary"),
            Button("4. Run Main Processing Pipeline", id="btn_pipe", variant="warning"),
            Button("5. Run Rescue Pipeline", id="btn_rescue", variant="warning"),
            Button("Quit", id="btn_quit", variant="error"),
            id="menu_container"
        )

class FaceFlowApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #menu_container {
        width: 60;
        height: auto;
        border: solid green;
        padding: 2 4;
        background: $surface;
    }
    #title {
        text-align: center;
        text-style: bold;
        width: 100%;
        margin-bottom: 1;
        color: $accent;
    }
    #subtitle {
        text-align: center;
        width: 100%;
        margin-bottom: 2;
    }
    Button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield MainMenu()
        yield Footer()

    def run_script(self, script_name: str):
        def _run():
            print(f"\n--- Running {script_name} ---")
            try:
                subprocess.run([sys.executable, script_name])
            except Exception as e:
                print(f"Error: {e}")
            print(f"\n--- Finished {script_name} ---")
            input("Press Enter to return to the menu...")

        with self.suspend():
            _run()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn_yt":
            self.run_script("downloader.py")
        elif button_id == "btn_tk1":
            self.run_script("tiktok_downloader.py")
        elif button_id == "btn_tk2":
            self.run_script("tiktok_downloader_v2.py")
        elif button_id == "btn_pipe":
            self.run_script("pipeline.py")
        elif button_id == "btn_rescue":
            self.run_script("rescue_multiface.py")
        elif button_id == "btn_quit":
            self.exit()

if __name__ == "__main__":
    app = FaceFlowApp()
    app.run()
