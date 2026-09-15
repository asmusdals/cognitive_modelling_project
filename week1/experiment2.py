"""Experiment 2 for the facial-feature encoding project (task 7).

Experiment 1 collected attractiveness ratings for real faces, and tasks 4-6
turned those ratings into a linear encoding model and a set of synthetic faces
built to have known ratings. Experiment 2 closes the loop: the synthetic faces
are shown to the test participants, who rate them on the same 1-5 scale. If the
model captures what the participants mean by attractiveness, the ratings they
give should follow the target ratings the faces were generated from. Task 8
(analyze_experiment2.py) measures that correlation.

The stimuli are the in-range set from task 6, not the requested set from task 5,
because four of the requested ratings fall outside the range the model actually
predicts for real faces and are therefore pure extrapolation. As task 7 asks,
every face is presented at least ten times in random order, so each face ends up
with a distribution of ratings rather than a single noisy trial.
"""

import csv
import random
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "2627yearmale"
SYNTHETIC_DIR = ROOT / "data/synthetic_faces" / DATASET_NAME
STIMULUS_SET = "in_range"
STIMULUS_DIR = SYNTHETIC_DIR / f"stimuli_{STIMULUS_SET}"
MANIFEST_FILE = SYNTHETIC_DIR / "synthetic_faces.csv"
DATA_DIR = ROOT / "data/experiment2" / DATASET_NAME

# Task 7 asks for every synthetic image to be presented at least 10 times.
REPEATS = 10


def load_stimuli():
    """Return the stimulus paths together with the rating each face was built for."""
    if not MANIFEST_FILE.exists():
        raise SystemExit(
            f"{MANIFEST_FILE} blev ikke fundet. Kør week1/synthetic_faces.py først."
        )

    with MANIFEST_FILE.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["set"] == STIMULUS_SET]

    if not rows:
        raise SystemExit(f"Manifestet indeholder ingen billeder i sættet '{STIMULUS_SET}'.")

    stimuli = []
    for row in rows:
        path = STIMULUS_DIR / row["filename"]
        if not path.exists():
            raise SystemExit(f"{path} blev ikke fundet. Kør week1/synthetic_faces.py igen.")
        # The filename is rounded to two decimals; the exact value is what the
        # image was actually constructed from, and what task 8 correlates against.
        stimuli.append((path, float(row["target_rating_exact"])))

    return sorted(stimuli, key=lambda stimulus: stimulus[1])


class Experiment:
    def __init__(self, window, stimuli, output_file):
        self.window = window
        self.output_file = output_file
        self.stimuli = stimuli
        self.ratings = {path.name: [] for path, _ in stimuli}
        self.trials = [path for path, _ in stimuli] * REPEATS
        random.shuffle(self.trials)
        while any(a == b for a, b in zip(self.trials, self.trials[1:])):
            random.shuffle(self.trials)
        self.trial_number = 0
        self.accepting_rating = False

        window.title("Experiment 2 – Syntetiske ansigter")
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
        self.show_start()

    def show_start(self):
        self.title.config(text="Vurdering af attraktivitet")
        self.instructions.config(
            text=(
                f"Du vil se {len(self.stimuli)} computergenererede ansigter, som hver "
                f"vises {REPEATS} gange i tilfældig rækkefølge.\n\nAnsigterne er slørede "
                "og ligner hinanden. Vurder dem alligevel med samme skala som før:\n"
                "1 = meget lidt attraktiv    2    3    4    5 = meget attraktiv\n\n"
                "Svar ud fra dit umiddelbare indtryk. Tryk MELLEMRUM for at begynde."
            )
        )
        self.progress.config(text="")

    def show_trial(self):
        path = self.trials[self.trial_number]
        with Image.open(path) as image:
            image.thumbnail((520, 520), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image.copy())
        self.image_label.config(image=self.photo)
        self.title.config(text="Hvor attraktivt er ansigtet?")
        self.instructions.config(
            text="1 = meget lidt attraktiv    2    3    4    5 = meget attraktiv"
        )
        self.progress.config(
            text=f"Billede {self.trial_number + 1} af {len(self.trials)}"
        )
        self.accepting_rating = True

    def handle_key(self, event):
        if self.trial_number == 0 and not self.accepting_rating and event.keysym == "space":
            self.show_trial()
        elif self.accepting_rating and event.char in {"1", "2", "3", "4", "5"}:
            self.accepting_rating = False
            filename = self.trials[self.trial_number].name
            self.ratings[filename].append(int(event.char))
            self.save_data()
            self.trial_number += 1
            if self.trial_number < len(self.trials):
                self.window.after(150, self.show_trial)
            else:
                self.finish()
        elif event.keysym == "Escape":
            self.quit_experiment()

    def save_data(self):
        with self.output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["filename", "target_rating"]
                + [f"rating_{number}" for number in range(1, REPEATS + 1)]
            )
            for path, target in self.stimuli:
                values = self.ratings[path.name]
                writer.writerow(
                    [path.name, f"{target:.6f}"]
                    + values
                    + [""] * (REPEATS - len(values))
                )

    def finish(self):
        self.image_label.config(image="")
        self.title.config(text="Tak for din deltagelse")
        self.instructions.config(text=f"Dine svar er gemt i:\n{self.output_file}")
        self.progress.config(text="Du kan nu lukke vinduet.")

    def quit_experiment(self):
        if self.trial_number == len(self.trials) or messagebox.askyesno(
            "Afslut?", "Forsøget er ikke færdigt. Vil du afslutte? Data gemmes delvist."
        ):
            self.window.destroy()


def main():
    stimuli = load_stimuli()

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
    Experiment(window, stimuli, output_file)
    window.mainloop()


if __name__ == "__main__":
    main()
