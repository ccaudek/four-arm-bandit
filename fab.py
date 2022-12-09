from __future__ import division
from psychopy import visual, data, event, core, gui
import numpy as np
from numpy import random
import os
import glob
import math
import pandas
import time as ts
import itertools

from fab_functions import show_fixation, get_bitmaps, get_img_chosen_by_subj, get_number_of_img_chosen_by_subj, get_payoff_matrix, draw_stim_images, get_reward_prob_for_chosen_image


# Declare primary task parameters.
params = {
    # Declare stimulus and response parameters
    "n_trials": 192,            # number of trials in this session
    "resp_keys": ["f", "g", "h", "j", "escape"],
    # Keys to be used for responses. f: upper-left; g: lower-left;
    # h: lower-right; j: upper_right.
    'img_dir': "./pics/",    # directory containing image stimluli
    "image_suffix": ".jpg",   # images will be selected randomly (without replacement) from all files in imageDir that end in imageSuffix.
# declare display parameters
    "fullscr": True,       # run in full screen mode?
    "monitor": "testMonitor",
    "colorSpace": "rgb",
    "screenToShow": 0,        # display on primary screen (0) or secondary (1)?
    "screenColor": [0, 0, 0] # in rgb255 space: (r,g,b) all between 0 and 255
}

# Make data file.
info = {"ppnr": 0, "Name": "", "Session": 0}
info["Name"] = "Corrado"

already_exists = True
while already_exists:
    myDlg = gui.DlgFromDict(dictionary = info,
                            title = "Probabilistic reversal learning task"
                        )
    directory_to_write_to = "/Users/corrado/Desktop/4armsbandit/data" + "/" + str(info["ppnr"])
    if not os.path.isdir(directory_to_write_to):
        os.mkdir(directory_to_write_to)
    file_name = directory_to_write_to + "/Prob_reversal_learning_task_subject_" + str(info["ppnr"]) + "_Session_" + str(info["Session"]) + "_data"
    if not os.path.isfile(file_name+".tsv"):
        already_exists = False
    else:
        myDlg2 = gui.Dlg(title = "Error")
        myDlg2.addText("Try another participant number")
        myDlg2.show()
print("OK, let's get started!")

thisExp = data.ExperimentHandler(dataFileName = file_name, extraInfo = info)
my_clock = core.Clock()

# Create a window.
window = visual.Window(fullscr = params["fullscr"],
                       monitor = params["monitor"],
                       color = params["screenColor"],
                       colorSpace = params["colorSpace"]
                )
print('window_ok')

fixation = visual.TextStim(window, text = ("+"))

neg_fb = visual.TextStim(window, text = ("- 10 punti"))
pos_fb = visual.TextStim(window, text = ("+ 10 punti"))
too_late = visual.TextStim(window, text = ("Rispondi più velocemente!"))
feedback = [neg_fb, pos_fb, too_late]
print('graphical_elements_ok')

welcome = visual.TextStim(
    window,
    text = ("Ciao {}!\n\n" +
            "Per proseguire premi la barra spaziatrice.\n\n").format(str(info["Name"])),
            wrapWidth = 50,
            units = "deg"
        )

# Within-subjects design.
TrialList = [] # assignment to an empty lis
trials = data.TrialHandler(trialList = TrialList, nReps = 1,
                           method = "sequential"
                    )
thisExp.addLoop(trials)

# Initialize data.
response = 0
corr = 0
fb = 2
points = 0
print('initialization_ok')

# Get stimulus files.
# img_dir = "/Users/corrado/OneDrive - unifi.it/programming/py/PieterV_public-master/in_progress/pics/"
img_dir = "./pics/"

# Get all files in <imageDir> that end in .<imageSuffix>.
# all_images = glob.glob(img_dir+"*"+params["image_suffix"])
all_images = [
    './pics/im1.jpg',
    './pics/im2.jpg',
    './pics/im3.jpg',
    './pics/im4.jpg'
    ]
# This means that all_images[0] is im1.jpg and so on. Then we have to establish
# where such image should be drawn.
print('%d images loaded from %s' % (len(all_images), params["img_dir"]))
print(all_images)

# The spatial locations are coded in clockwise orientation as follows:
# 0 1
# 3 2

# stim_position = [
#     [0, 1, 2, 3],
#     [2, 1, 3, 0],
#     [0, 1, 2, 3],
#     [3, 0, 1, 2]
#     ]

# The numbers identify the images with the following coding scheme:
# 0: image "a"; 1: image "b"; 2: image "c"; 3: image "d".
# The order of the elements of the vector (first element, second element, ...)
# identifies the spatial position in which the image is located. The first
# element refers to the upper-left position, the second element refers to the
# upper-right position, the third element refers to the lower-right position,
# the forth element refers to the lower-left position.
# The numbers in the row [2, 1, 3, 0], for example, mean that
# img "c" is located in the upper-left position;
# img "b" is located in the upper-right position;
# img "d" is located in the lower-right position;
# img "a" is located in the lower-left position.
M = list(itertools.permutations([0, 1, 2, 3]))
stim_position = np.tile(M, (8, 1))
# Randomize order.
random.shuffle(stim_position)
# print(stim_position)

payoff = get_payoff_matrix(params["n_trials"])
# print(payoff)

# Welcome screen.
welcome.draw()
window.flip()

# Start experiment.
event.waitKeys(keyList = "space")

# Trial loop.
for trial in range(params["n_trials"]):

    # Draw fixation cross.
    show_fixation(window, fixation)

    # Draw stimulus images.

    draw_stim_images(window, all_images, stim_position, trial)

    # Clear buffer for keyboard input.
    event.clearEvents(eventType = "keyboard")

    # Start clock.
    my_clock.reset()

    # Get subject's response.
    response = event.waitKeys(keyList = params["resp_keys"])

    # Get RT.
    rt = my_clock.getTime()

    # Get probability of reward for the image chosen by the subject.
    reward_prob_for_chosen_image = get_reward_prob_for_chosen_image(
        response[0], all_images, stim_position, payoff, trial
    )

    # Provide feedback.
    if rt > 1.5:
        fb = 2
    else:
        if np.random.random() >= reward_prob_for_chosen_image:
            fb = 0 # punishment
        else:
            fb = 1 # reward

    if fb == 1:
        points += 10
    elif fb == 0:
        points -= 10
    else:
        points += 0

    # Exit trial loop if ESC.
    if response[0] == "escape":
        thisExp.saveAsWideText(file_name, appendFile=False)
        thisExp.abort()
        core.quit()

    # Record trial number, keypress, RT.
    # Add the current time.
    date_str = ts.strftime("%b_%d_%H:%M:%S", ts.localtime())
    trials.addData("Time", date_str)
    trials.addData("TrialNumber", trial)
    trials.addData("Response", response[0])
    # trials.addData("ImageChosenByParticipant", img_chosen)
    # trials.addData("NumberImageChosenByParticipant", num_img_chosen)
    trials.addData("RT", rt)
    trials.addData("Feedback", fb)
    trials.addData("Points", points)
    # trials.addData("IndexOfChosenImgWithinRow", index_of_chosen_img_within_row)

    # New line in data file.
    thisExp.nextEntry()

    feedback[fb].draw()
    window.flip()
    core.wait(1)

    # TODO: show feedback for 0.5 s. Then add a blanck screen with random
    # duration between 0.5 s and 1.0 s.

# Close the window after trial loop.
window.close()

# Close PsychoPy.
core.quit()
