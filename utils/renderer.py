import os
import subprocess

class Renderer:
    def __init__(self, envs, model, output_file="output.txt", record=False, nb_frames=0):
        self.envs = envs
        self.env = envs
        self.model = model
        self.current_frame = self._get_current_frame()
        self.paused = False
        self.running = True
        self.record = record
        self.nb_frames = nb_frames
        self.output_file = output_file  # Datei, in die geschrieben wird

        # Initialisieren und Datei leeren
        with open(self.output_file, "w") as file:
            file.write("Frame\tX\tY\tVelX\tVelY\tAngle\tAngularVel\tLeg1\tLeg2\n")  # Header hinzufügen

    def run(self):
        obs = self.envs.reset()
        frame_counter = 0
        while self.running:
            # Modellvorhersage und Umgebungsinteraktion
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.envs.step(action)
            self.current_frame = self._get_current_frame()

            # Ausgabe in Datei schreiben
            self._write_to_file(frame_counter)

            # Prüfen, ob beide Beine auf dem Boden sind
            if self._check_landing_condition(self.current_frame):
                print("Both legs have landed. Stopping rendering.")
                self.running = False  # Rendering beenden

            if done:
                obs = self.envs.reset()

            frame_counter += 1

        # Animations-Skript ausführen
        self._run_animation_script()

    def _get_current_frame(self):
        # Greifen Sie auf die erste Umgebung innerhalb von DummyVecEnv zu
        actual_env = self.env.envs[0] if hasattr(self.env, 'envs') else self.env
        if hasattr(actual_env, '_obj_obs'):
            return actual_env._obj_obs
        else:
            raise AttributeError("The actual environment does not have the attribute '_obj_obs'")

    def _write_to_file(self, frame_counter):
        """
        Schreibt den aktuellen Frame in die Ausgabedatei.
        """
        with open(self.output_file, "a") as file:
            formatted_frame = "\t".join(f"{value:.5f}" for value in self.current_frame[:6]) + \
                              f"\t{int(self.current_frame[6])}\t{int(self.current_frame[7])}"
            file.write(f"{frame_counter}\t{formatted_frame}\n")

    def _check_landing_condition(self, frame):
        """
        Überprüfen, ob beide Beine den Boden berühren.
        Die letzten beiden Werte des Arrays sind Booleans für die Beine.
        """
        if len(frame) >= 8:  # Sicherstellen, dass das Array korrekt ist
            return frame[-1] and frame[-2]  # Beide Beine auf dem Boden
        return False

    def _run_animation_script(self):
        """
        Führt das Animationsskript aus, nachdem die Simulation abgeschlossen ist.
        """
        animation_script_path = r"C:\Studium_TU_Darmstadt\Master\1. Semester\KI Praktikum\WorkingEnv\KI-Praktikum-ESA\lunar_lander_animation.py"
        
        if not os.path.isfile(animation_script_path):
            print(f"Animation script not found at: {animation_script_path}")
            return
        
        try:
            print("Starting animation script...")
            subprocess.run(["python", animation_script_path], check=True)
            print("Animation script completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error while running animation script: {e}")

