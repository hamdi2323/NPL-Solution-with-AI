"""
===============================================================
Python script to demonstrate a document classification and topic modelling
===============================================================
This program demonstrate text-to-speech using pyttsx3 package

Flow:
    1.Read in a text file(passage.txt)
    2.Process the text in a python program
    3.Text-to-Speech Engine generates audio
    4.Output an audio file (travel_output.wav)

Input/Output file location:
    files/passage.txt
    files/travel_output.wav
Requirements:
    !pip install pyttsx3
Author:Xamdi Salaad
Date: 04-06-2026
"""
#------------------------------------------------------------
#0.Import required modules
#-----------------------------------------------------------
import logging # (optional to show what is happening at what time)
import pyttsx3
import sys
from pathlib import Path

import warnings

#suppress warning for cleaner output demom
warnings.filterwarnings("ignore")

#------------------------------------------------------------
#1.Optional Logging Configuration
#-----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ %(levelname)s]  %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)

#------------------------------------------------------------
#2. Constants
#----------------------------------------------------------
TEXT_FILE: Path = Path("../files/passage.txt")
OUTPUT_FILE: Path = Path("../files/travel_output.wav")

SPEAKING_RATE: int = 150  #words per minute (normal conversational pace)
VOLUME: float = 1.0       #full volume (range: 0.0 - 1.0)

#------------------------------------------------------------
#3.Functions
#-----------------------------------------------------------
def load_text(file_path: Path) -> str:

    #optional logging
    logger.info(f"Loading text file {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"File {file_path} does not exist"
            f"\nPlease ensure the 'passage.txt' exists in the files folder "
        )

    text: str = file_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("Text file cannot be empty"
                         f"\nPlease add some text.")

    logger.info(f"Successfully loaded text from {file_path}...")
    return text

def initialise_engine() -> pyttsx3.Engine:
    #Optional logging
    logger.info("Initialising pyttsx3 engine...")

    try:
        engine : pyttsx3.Engine = pyttsx3.init()
    except Exception as exec:
        raise RuntimeError(f"Failed to initialise pyttsx3 engine."
                           f"\nPlease ensure that pyttsx3 engine is installed.") from exec

    #set a moderate speaking rate (wmp)
    engine.setProperty('rate', SPEAKING_RATE)
    #set the volume to max
    engine.setProperty('volume', VOLUME)
    #optional logging
    logger.info(
        f"Engine ready | Rate: {SPEAKING_RATE} | Volume: {VOLUME:.3f}"
    )
    return engine

def save_audio(engine: pyttsx3.Engine, text:str, output_path:Path) -> None:

    #optional logging
    logger.info(f"Saving audio to {output_path}...")  # Fixed: Changed output_file to output_path

    #ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    #pyttsx3 save audio by queuing the text and then running the engine
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    if not output_path.exists():
        raise IOError(  # Fixed: Changed IOErrorr to IOError
            f"Audio file was not created at: {output_path}"
            f"\nPlease ensure that pyttsx3 has permission to write to the 'files' folder"
        )
    #optional logging
    logger.info(f"Audio file saved at: {output_path}")

def speak_text(engine:pyttsx3.Engine, text:str) -> None:
    #optional logging
    logger.info("Speaking text aloud...")
    engine.say(text)  # Fixed: Added this line to actually speak the text
    engine.runAndWait()
    logger.info("Audio playback complete")

#------------------------------------------------------------
#4.Main execution Function
#-----------------------------------------------------------
def main() -> None:
    print()
    print("-"*30)
    print(" TEXT TO SPEECH DEMONSTRATION ")
    print("-"*30)
    print()

    # -------------------------------
    # Step I: Load the text
    # ------------------------------
    print("Loading text...")
    try:
        text = load_text(TEXT_FILE)
    except (FileNotFoundError, ValueError) as exec:
        #optional logging
        logger.error(f"Could not load text {exec}")
        sys.exit(1)

    # -----------------------------------
    # Step II: Display the text on screen
    # -----------------------------------
    print()
    print("-"*40)
    print(" PASSAGE  ")
    print("-"*40)
    print()
    print(text)  # Fixed: Added this line to actually display the text
    print()

    # ------------------------------------
    # Step III: Initialise the TTS engine
    # -----------------------------------
    try:
        engine: pyttsx3.Engine = initialise_engine()
    except RuntimeError as exec:
        #optional logger
        logger.error(f"Engine initialization failed: {exec}")
        sys.exit(1)

    # --------------------------------------
    # Step IV: Save the audio to a .wav file
    # ---------------------------------------
    print("Generating audio file...")
    try:
        save_audio(engine, text, OUTPUT_FILE)
    except IOError as exec:
        #optional logger
        logger.error(f"Could not save audio file {exec}")
        sys.exit(1)
    print(f"Audio saved to {OUTPUT_FILE}\n")

    # -------------------------------
    # Step V: Speak the text aloud
    # ------------------------------
    print("Speaking the text now - please ensure your speakers are on....\n")
    try:
        speak_text(engine, text)
    except Exception as exec:
        logger.error(f"Could not speak text {exec}")
        sys.exit(1)
    print("\nEnd of demonstration")
    print("-"*40)

# ------------------------------------------------------------
# 5.Run the script by invoking its main function
# -----------------------------------------------------------
if __name__ == "__main__":
    main()