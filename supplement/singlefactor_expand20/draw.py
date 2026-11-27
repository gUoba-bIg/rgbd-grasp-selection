"""Replay the original read-constrained draw. No method results are accessible to this process."""
from paths import *
import blind_sampling_check as frozen

def main():
    # The frozen callback permits one output only; all other file opens are limited to
    # the original observation plan and 600 pairs of pre-E pools and ranking files.
    need(sha(Path(frozen.__file__)) == "783821801633809034c49ecc9eba3a162fdcd657418b3741089c9ae5c5b650f5", "frozen sampling function changed")
    frozen.REPLAY = OUT / "preregistration/replayed_frame.json"
    def save(_original_name, record):
        return write("preregistration/replayed_frame.json", record)
    frozen.write = save
    frozen.draw()
if __name__ == "__main__":
    main()
