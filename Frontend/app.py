# streamlit_app.py
import streamlit as st
import sys
import os
import re

# Pfad zum Backend und Modellverzeichnis hinzufügen
sys.path.append(os.path.abspath('../backend'))
from Interpolation import main

# Titel
st.title('Image Generation from Seeds')


def sort_numerically(files):
    def extract_number(filename):
        s = re.findall(r'\d+', filename)
        return int(s[0]) if s else -1

    return sorted(files, key=extract_number)


# Pfad zum Modellverzeichnis
model_dir = os.path.abspath('../../Model_128x128_end')
# Liste der Modelldateien erhalten und numerisch sortieren
model_files = sort_numerically(os.listdir(model_dir))
model_selection = st.selectbox('Select a model', model_files)

# Benutzereingaben für seeds
seed1 = st.number_input('Seed 1', min_value=0, max_value=10000, value=1)
seed2 = st.number_input('Seed 2', min_value=0, max_value=10000, value=2)
steps = 50  # Oder Benutzer lassen wählen
gif_fps = 20  # Basierend auf gewünschter Geschwindigkeit anpassen

if st.button('Generate GIF'):
    # Vollständigen Pfad zum ausgewählten Modell erstellen
    model_path = os.path.join(model_dir, model_selection)
    gif_path = main(seed1, seed2, steps, gif_fps, model_path)  # main muss angepasst werden, um model_path zu akzeptieren
    st.image(gif_path, caption=f'GIF from seed {seed1} to {seed2}', use_column_width=True)
