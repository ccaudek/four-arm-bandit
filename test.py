# Import required modules
from psychopy import visual, core, event
import numpy as np

# Initialize variables
nTrials = 100  # Number of trials
pCorrect = 0.8  # Probability of correct response on each trial

# Initialize arrays to store results
response = np.zeros(nTrials)  # Response on each trial
correct = np.zeros(nTrials)  # Correctness of response on each trial

# Initialize variables for probabilistic reversal
reversal = False  # Whether reversal has occurred
stage = 1  # Current stage of learning

# Create a window to display the task
win = visual.Window(size = [800,600], color = [1,1,1])

# Create stimuli for the task
stimA = visual.TextStim(win, text = "A", color = [0,0,0])
stimB = visual.TextStim(win, text = "B", color = [0,0,0])

# Perform probabilistic reversal learning
for i in range(nTrials):
  # Draw stimuli on the screen
  if stage == 1:
    stimA.draw()
    stimB.draw()
  elif stage == 2:
    stimB.draw()
    stimA.draw()

  # Update the screen
  win.flip()

  # Wait for response
  response[i] = event.waitKeys(keyList = ["a","b"])

  # Determine whether response was correct
  if stage == 1 and response[i] == "a" or stage == 2 and response[i] == "b":
    correct[i] = 1

  # Check for reversal
  if not reversal and sum(correct[max(0,i-19):i+1]) < 15:
    reversal = True
    stage = 2

# Close the window
win.close()
