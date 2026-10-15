"""Run once:  python download_models.py   (also happens automatically on first run)."""
import utils

if __name__ == "__main__":
    print("Model ready at:", utils.ensure_landmark_model())
