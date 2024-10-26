import pyniryo
from pyniryo2 import *
import random
import pyniryo2
import time
import sys

robot = NiryoRobot("169.254.200.200")
ros_instance = NiryoRos("169.254.200.200")

sound_files = [
        ("agent turn1.wav", "voice/agent turn1.wav"),
        ("agent turn3.wav", "voice/agent turn3.wav"),
        ("agent turn4.wav", "voice/agent turn4.wav"),
        ("agent win1.wav", "voice/agent win1.wav"),
        ("draw 1.wav", "voice/draw 1.wav"),
        ("hello1.wav", "voice/hello1.wav"),
        ("hello2.wav", "voice/hello2.wav"),
        ("hello3.wav", "voice/hello3.wav"),
        ("player turn2.wav", "voice/player turn2.wav"),
        ("player turn.wav", "voice/player turn.wav"),
        ("player win1.wav", "voice/player win1.wav"),
        ("player win3.wav", "voice/player win3.wav")
    ]

sound_names = sound_names = [name for name, _ in sound_files]
sound_dict = dict(sound_files)


def savesounds(ros_instance, sound_files):
    sound = pyniryo2.Sound(ros_instance)

    for sound_name, sound_path in sound_files:
        try:
            sound.save(sound_name, sound_path)
            print(f"Successfully saved: {sound_name}")
        except Exception as e:
            print(f"Error saving {sound_name}: {str(e)}")

# Assuming you have already created the ros_instance
# ros_instance = ... (your ROS instance creation code here)

# Call the function
# savesounds(ros_instance, sound_files)

def play_random_sound(robot, sound_choices):
    """
    Play a random sound from the given choices.
    
    :param robot: NiryoRobot instance
    :param sound_choices: List of sound names to choose from
    """
    # Filter sound_choices to only include valid sound names
    valid_choices = [choice for choice in sound_choices if choice in sound_dict]
    
    if not valid_choices:
        print("No valid sound choices provided.")
        return
    
    # Randomly select a sound from the valid choices
    chosen_sound = random.choice(valid_choices)
    
    try:
        # Play the chosen sound
        robot.sound.play(chosen_sound, wait_end=True)
        # Wait for the sound to finish
        # while robot.sound.is_playing():
        #     time.sleep(0.1)
        print(f"Playing sound: {chosen_sound}")
        # robot.sound.stop()
    except Exception as e:
        print(f"Error playing sound {chosen_sound}: {str(e)}")


def startsound():
    play_random_sound(robot, ["hello1.wav"])
    time.sleep(0.5)
    return

def ingamesound():
    play_random_sound(robot, ["hello2.wav", "hello3.wav"])
    time.sleep(0.5)
    return

def endgamesound():
    play_random_sound(robot, ["agent win1.wav", "player win1.wav", "player win3.wav"])
    time.sleep(0.5)
    return

def allsounds():
    play_random_sound(robot, sound_names)
    time.sleep(0.5)
    return
    
# try:
#     print('Starting to play sounds')
#     for i in range(2):
#         ingamesound()
#         allsounds()
#         print(f'Finished playing sound: {i+1}')
        
#     print('All sounds played successfully')
# except Exception as e:
#     print(f"An error occurred: {e}")

        # Ensure the robot connection is closed properly
        
        
#     print("Program ending")
#     exit()  # Exit with success code
    
# if __name__ == "__main__":
#     main()
