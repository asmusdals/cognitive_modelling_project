"""Experiment 3 for the facial-feature encoding project (task 8).

Experiment 2 asked whether the model orders the synthetic faces the way the
participants do. Experiment 3 asks something stronger: whether the continuum the
model defines behaves like a perceptual dimension. If it does, staring at a face
from one end of the continuum should produce an after-effect, so that a face
near the neutral point afterwards looks shifted towards the opposite end.

Each trial adapts the observer to one endpoint of the continuum for ADAPT_SECONDS
and then flashes one of the three near-neutral test faces for TEST_SECONDS,
after which the observer rates it on the same 1-5 scale as in the previous
experiments. Both endpoints are used as adapting stimuli in separate conditions,
which with three test faces gives the six conditions the task asks for. A
fixation point is drawn in the centre of every stimulus so that the observer
keeps the adapting and test faces on the same part of the retina, which is where
the after-effect lives.

The trials are blocked by adapting face rather than intermixed. Adaptation
carries over from one trial to the next, and alternating between opposite
adapters would let the two conditions cancel each other out. The order of the
two blocks is randomised per participant so that any drift over the session does
not systematically favour one adapter.
"""

import csv
import random
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from PIL import Image, ImageDraw, ImageTk


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
SYNTHETIC_DIR = ROOT / "data/synthetic_faces" / DATASET_NAME
MANIFEST_FILE = SYNTHETIC_DIR / "synthetic_faces.csv"
STIMULUS_DIRS = {
    "exp3_adapt": SYNTHETIC_DIR / "stimuli_experiment3_adapt",
    "exp3_test": SYNTHETIC_DIR / "stimuli_experiment3_test",
}
DATA_DIR = ROOT / "data/experiment3" / DATASET_NAME

# The task specifies 20-30 seconds of adaptation followed by a 0.5-1.0 second
# test stimulus, so that the test face is judged while adaptation is still
# strong. Six conditions repeated three times give 18 trials, about 8 minutes.
ADAPT_SECONDS = 20
TEST_SECONDS = 0.75
REPEATS_PER_CONDITION = 3
# A blank field with only the fixation point separates the trials. Without it the
# adapting face simply stays on screen from one trial to the next, and the
# observer cannot see where one trial ends and the next begins. It sits before
# the adaptation, never between the adapting and the test face.
INTER_TRIAL_SECONDS = 0.8
INTER_TRIAL_GREY = (128, 128, 128)

FIXATION_RADIUS = 3
FIXATION_COLOUR = "#d92b2b"


def load_stimuli() -> tuple[list[dict], list[dict]]:
    """Return the two adapting faces and the three test faces from the manifest."""
    if not MANIFEST_FILE.exists():
        raise SystemExit(
            f"{MANIFEST_FILE} blev ikke fundet. Kør week1/synthetic_faces.py først."
        )

    with MANIFEST_FILE.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    stimuli: dict[str, list[dict]] = {}
    for set_name, directory in STIMULUS_DIRS.items():
        entries = []
        for row in rows:
            if row["set"] != set_name:
                continue
            path = directory / row["filename"]
            if not path.exists():
                raise SystemExit(
                    f"{path} blev ikke fundet. Kør week1/synthetic_faces.py igen."
                )
            entries.append(
                {
                    "name": row["filename"],
                    "path": path,
                    # The filename carries two decimals; this is the exact rating
                    # the image was constructed for.
                    "rating": float(row["target_rating_exact"]),
                }
            )
        if not entries:
            raise SystemExit(
                f"Manifestet indeholder ingen billeder i sættet '{set_name}'. "
                "Kør week1/synthetic_faces.py igen."
            )
        stimuli[set_name] = sorted(entries, key=lambda entry: entry["rating"])

    adapt_stimuli = stimuli["exp3_adapt"]
    if len(adapt_stimuli) != 2:
        raise SystemExit(
            f"Der skal være præcis to adaptere, men manifestet har {len(adapt_stimuli)}."
        )
    adapt_stimuli[0]["condition"] = "lav"
    adapt_stimuli[-1]["condition"] = "høj"

    return adapt_stimuli, stimuli["exp3_test"]


def build_blocks(adapt_stimuli: list[dict], test_stimuli: list[dict]) -> list[dict]:
    """One block per adapting face, with the test faces repeated in random order."""
    blocks = []
    for adapt in adapt_stimuli:
        trials = []
        for repetition in range(1, REPEATS_PER_CONDITION + 1):
            order = list(test_stimuli)
            random.shuffle(order)
            trials.extend({"repetition": repetition, "test": test} for test in order)
        blocks.append({"adapt": adapt, "trials": trials})

    random.shuffle(blocks)
    return blocks


def draw_fixation(picture: Image.Image) -> None:
    """Draw the point the observer must gaze at in the centre of the image."""
    draw = ImageDraw.Draw(picture)
    centre_x, centre_y = picture.width / 2, picture.height / 2
    draw.ellipse(
        [
            centre_x - FIXATION_RADIUS,
            centre_y - FIXATION_RADIUS,
            centre_x + FIXATION_RADIUS,
            centre_y + FIXATION_RADIUS,
        ],
        fill=FIXATION_COLOUR,
        outline="white",
    )


def load_with_fixation(path: Path) -> Image.Image:
    """Load a stimulus and draw the fixation point in its centre."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((520, 520), Image.Resampling.LANCZOS)
        picture = image.copy()

    draw_fixation(picture)
    return picture


def make_fixation_screen(size: tuple[int, int]) -> Image.Image:
    """A blank field with only the fixation point, shown between the trials."""
    picture = Image.new("RGB", size, INTER_TRIAL_GREY)
    draw_fixation(picture)
    return picture


class Experiment:
    def __init__(self, window, adapt_stimuli, test_stimuli, output_file):
        self.window = window
        self.output_file = output_file
        self.blocks = build_blocks(adapt_stimuli, test_stimuli)
        self.n_trials = sum(len(block["trials"]) for block in self.blocks)
        self.block_index = 0
        self.trial_index = 0
        self.completed = 0
        self.responses = []
        self.accepting_rating = False
        self.waiting_for_space = False
        self.pending = None

        # The test stimulus is only shown for a fraction of a second, so every
        # image is converted once here instead of being read from disk mid-trial.
        pictures = {
            stimulus["name"]: load_with_fixation(stimulus["path"])
            for stimulus in adapt_stimuli + test_stimuli
        }
        self.photos = {
            name: ImageTk.PhotoImage(picture) for name, picture in pictures.items()
        }
        self.fixation_screen = ImageTk.PhotoImage(
            make_fixation_screen(next(iter(pictures.values())).size)
        )

        window.title("Experiment 3 – Adaptation")
        window.geometry("750x750")
        window.configure(bg="white")
        window.bind("<Key>", self.handle_key)
        window.protocol("WM_DELETE_WINDOW", self.quit_experiment)

        self.title = tk.Label(window, font=("Arial", 22, "bold"), bg="white")
        self.title.pack(pady=(35, 10))
        self.image_label = tk.Label(window, bg="white")
        self.image_label.pack(expand=True)
        self.instructions = tk.Label(
            window, font=("Arial", 16), bg="white", justify="center", wraplength=680
        )
        self.instructions.pack(pady=20)
        self.progress = tk.Label(window, font=("Arial", 12), bg="white", fg="gray")
        self.progress.pack(pady=(0, 25))
        self.show_block_intro()

    @property
    def block(self):
        return self.blocks[self.block_index]

    @property
    def trial(self):
        return self.block["trials"][self.trial_index]

    def schedule(self, milliseconds, callback):
        self.pending = self.window.after(milliseconds, callback)

    def show_block_intro(self):
        self.image_label.config(image="")
        self.waiting_for_space = True
        self.title.config(text=f"Del {self.block_index + 1} af {len(self.blocks)}")
        self.instructions.config(
            text=(
                f"Du vil se et ansigt i {ADAPT_SECONDS} sekunder. Hold hele tiden "
                "blikket på den røde prik i midten.\n\nDerefter blinker et andet "
                "ansigt kort forbi. Vurder det ansigt med tasterne:\n"
                "1 = meget lidt attraktiv    2    3    4    5 = meget attraktiv\n\n"
                f"Det gentages {len(self.block['trials'])} gange i denne del. Det er "
                "med vilje, at det er det samme ansigt, du ser i de 20 sekunder hver "
                "gang, og de ansigter, du skal vurdere, ligner hinanden meget. Svar "
                "ud fra dit umiddelbare indtryk.\n\nTryk MELLEMRUM for at begynde."
            )
        )
        self.progress.config(text=f"{self.completed} af {self.n_trials} svar givet")

    def show_inter_trial(self):
        """Blank the screen briefly so the observer can see a new trial starting."""
        self.image_label.config(image=self.fixation_screen)
        self.title.config(text="")
        self.instructions.config(text="Bliv på prikken – næste ansigt kommer nu.")
        self.progress.config(text="")
        self.schedule(int(INTER_TRIAL_SECONDS * 1000), self.show_adapt)

    def show_adapt(self, seconds_left=None):
        if seconds_left is None:
            seconds_left = ADAPT_SECONDS
        # The same adapting face is shown on every trial in a block. That is
        # deliberate: the after-effect has to build up over the whole block.
        self.image_label.config(image=self.photos[self.block["adapt"]["name"]])
        self.title.config(text="Se på den røde prik")
        self.instructions.config(text="Hold blikket i midten af ansigtet.")
        self.progress.config(
            text=f"Runde {self.completed + 1} af {self.n_trials} – "
            f"{seconds_left} sekunder tilbage"
        )

        if seconds_left > 0:
            self.schedule(1000, lambda: self.show_adapt(seconds_left - 1))
        else:
            self.show_test()

    def show_test(self):
        # The test face replaces the adapting face with no gap in between, so
        # that it is judged while the observer is still adapted.
        self.image_label.config(image=self.photos[self.trial["test"]["name"]])
        self.title.config(text="")
        self.instructions.config(text="")
        self.progress.config(text="")
        self.schedule(int(TEST_SECONDS * 1000), self.ask_for_rating)

    def ask_for_rating(self):
        self.image_label.config(image="")
        self.title.config(text="Hvor attraktivt var ansigtet?")
        self.instructions.config(
            text="1 = meget lidt attraktiv    2    3    4    5 = meget attraktiv"
        )
        self.progress.config(
            text=f"Svar {self.completed + 1} af {self.n_trials}"
        )
        self.accepting_rating = True

    def handle_key(self, event):
        if self.waiting_for_space and event.keysym == "space":
            self.waiting_for_space = False
            self.show_adapt()
        elif self.accepting_rating and event.char in {"1", "2", "3", "4", "5"}:
            self.accepting_rating = False
            self.record_rating(int(event.char))
        elif event.keysym == "Escape":
            self.quit_experiment()

    def record_rating(self, rating):
        adapt = self.block["adapt"]
        test = self.trial["test"]
        self.responses.append(
            {
                "block": self.block_index + 1,
                "trial": self.completed + 1,
                "repetition": self.trial["repetition"],
                "adapt_condition": adapt["condition"],
                "adapt_filename": adapt["name"],
                "adapt_rating": f"{adapt['rating']:.6f}",
                "test_filename": test["name"],
                "test_rating": f"{test['rating']:.6f}",
                "rating": rating,
            }
        )
        self.save_data()
        self.completed += 1
        self.trial_index += 1

        if self.trial_index < len(self.block["trials"]):
            self.show_inter_trial()
        elif self.block_index + 1 < len(self.blocks):
            self.block_index += 1
            self.trial_index = 0
            self.show_block_intro()
        else:
            self.finish()

    def save_data(self):
        fieldnames = [
            "block",
            "trial",
            "repetition",
            "adapt_condition",
            "adapt_filename",
            "adapt_rating",
            "test_filename",
            "test_rating",
            "rating",
        ]
        with self.output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.responses)

    def finish(self):
        self.image_label.config(image="")
        self.title.config(text="Tak for din deltagelse")
        self.instructions.config(text=f"Dine svar er gemt i:\n{self.output_file}")
        self.progress.config(text="Du kan nu lukke vinduet.")

    def quit_experiment(self):
        if self.completed == self.n_trials or messagebox.askyesno(
            "Afslut?", "Forsøget er ikke færdigt. Vil du afslutte? Data gemmes delvist."
        ):
            if self.pending is not None:
                self.window.after_cancel(self.pending)
            self.window.destroy()


def main():
    adapt_stimuli, test_stimuli = load_stimuli()

    window = tk.Tk()
    window.withdraw()
    participant = simpledialog.askstring(
        "Deltager", "Indtast dit studienummer:", parent=window
    )
    if not participant or not re.fullmatch(r"[A-Za-z0-9_-]+", participant):
        messagebox.showerror(
            "Ugyldigt studienummer",
            "Brug kun bogstaver, tal, bindestreg og underscore.",
            parent=window,
        )
        window.destroy()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / f"{participant}.csv"
    if output_file.exists():
        messagebox.showerror(
            "Filen findes allerede",
            f"{output_file.name} findes allerede og bliver ikke overskrevet.",
            parent=window,
        )
        window.destroy()
        return

    window.deiconify()
    Experiment(window, adapt_stimuli, test_stimuli, output_file)
    window.mainloop()


if __name__ == "__main__":
    main()
