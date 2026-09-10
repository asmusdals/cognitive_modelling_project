import csv
import random
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/processed/26yearmale"
DATA_DIR = ROOT / "data/experiment1"


class Experiment:
    def __init__(self, window, images, participant, output_file):
        self.window = window
        self.participant = participant
        self.output_file = output_file
        self.ratings = {image.name: [] for image in images}
        self.image_count = len(images)
        self.trials = images * 2
        random.shuffle(self.trials)
        while any(a == b for a, b in zip(self.trials, self.trials[1:])):
            random.shuffle(self.trials)
        self.trial_number = 0
        self.accepting_rating = False

        window.title("Experiment 1 – Attractiveness ratings")
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
                f"Du vil se {self.image_count} ansigter, som hver vises to gange i tilfældig "
                "rækkefølge.\n\nVurder hvert ansigts attraktivitet med tasterne:\n"
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
        elif self.accepting_rating and event.char in "12345":
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
            writer.writerow(["filename", "rating_1", "rating_2"])
            for filename, values in sorted(self.ratings.items()):
                writer.writerow([filename] + values + [""] * (2 - len(values)))

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
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"Der blev ikke fundet nogen JPG-billeder i {IMAGE_DIR}.")

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
    Experiment(window, images, participant, output_file)
    window.mainloop()


if __name__ == "__main__":
    main()
