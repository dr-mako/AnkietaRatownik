from pathlib import Path


#Uruchom:
#python .\podglad_slk.py

path = Path("badanie urazowe_Myo.slk")

with path.open("r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= 80:
            break
        print(f"{i+1:03d}: {line.rstrip()}")