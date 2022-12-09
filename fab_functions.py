## fab_functions.py
##
## Project: Four-armed bandit
## Purpose: Functions to be used by the script fab.py
## Author: Corrado Caudek
## Date: 2020-11-20

## I created a matrix with n rows and 4 columns, where n is the number of
## trials. To generate this matrix I started from the vector [0 1 2 3], where
## the numbers indicate the images (0 denotes the file-name for a given image,
## 1 denotes the file-name for a second image, etc.). There are 4 bandits, so
## there must be 4 images. The index of the elements of this vector denotes the
## spatial position in which the image is drawn. So, the index 0 refers to the
## image that is located in the upper left corner; 1 refers to the image in
## the upper right corner; 2 lower right; 3 lower left.
## We must solve two problems: where to place each image in each trial, and to
## understand which image has been chosen by the subject in that trial. For the
## first problem, I created all the 24 permutations of the vector [0 1 2 3] --
## i.e., a matrix with 24 rows.  Then I glued together 8 of these matrices for
## total of 192 rows. Then I randomized the rows of such matrix.
## Each trial occurs within a for loop having 192 cycles. So, to decide how to
## draw the 4 images on the screen in the i-th trial, I select the i-th row of
## the matrix that I have created.  For example, [2 0 3 1]. This means that the
## image code by '2' should be drawn in the upper left corner, ans so on. Then
## I have to understand which image the subject has chosen.  I know that in the
## i-th trial the subject has pressed, for example, the key 'g'.  I know that
## 'g' means 'lower left'. Therefore I have to see which image has been drawn,
## in the i-th trial, in the lower left position.  The position 'lower left'
## corresponds to the row index == 3. The element of the i-th row that is in
## position 3 is '1'. This means that the subject has chosen the image coded by
## '1'.
## What remains to be determined is how to determine the probability of a
## positive a feedback for the chosen image in the i-th trial. This information
## is provided by the payoff matrix. The 192 x 4 payoff matrix contains the
## probabilities of a reward for each image in each of the 192 trials. I know
## which image has been chosen by the subject in the i-th trial. For example,
## the image 'c'. The payoff matrix reports the random walk of the probability
## of reward for the image 'a' in the first column (0), for the image 'b' in
## the second column (1), for 'c' in the third column (2), and for 'd' in the
## fourth column (3). Therefore, by knowing which image has been chosen by the
## subject in the i-th trial, I look for the probability associated to that
## image in the i-th row of the payoff matrix. For example, if the subject has
## chosen the image 'c', than the probability of reward is in the i-th row and
## the third column (indexed by '2') of the payoff matrix.


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


#-------------------------------------------------------------------------------
def draw_stim_images(window, all_images, stim_position, trial):
    """
    Draw the four stimulus images on the screen according to their position
    determined by the row (index "trial") of the matrix "stim_position". The
    names of the image files are in "all_images".
    """
    bitmap = get_bitmaps(window, all_images, stim_position, trial)

    bitmap[0].draw()
    bitmap[1].draw()
    bitmap[2].draw()
    bitmap[3].draw()

    window.flip()


#-------------------------------------------------------------------------------
def get_bitmaps(window, all_images, stim_position, trial):
    """
    Returns four bitmaps. Each one is associating to one of four positions of
    the graphical window. The stim_position matrix is used to associate each of
    the four stimulus images to a given position within the graphical window.
    Example: img_ul is bitmap that is positioned in the (-0.45, 0.45) location
    (upper-left corner). The image that is depicted in such bitmap is chosen
    according to the first element (0-th index) of the row of the stim_position
    matrix determined by the "trial" index. In the stim_position matrix, the
    images are code so that 0 corresponds to image "a", 1: image "b", 2: image
    "c", 3: image "d". For example, if the trial-th row of stim_position is
    [2 0 3 1], then the upper-left position will depict the image "c"; the
    image "a" will be positioned in upper-right position, the image "d" in the
    lower-right position, and the image "b" will be in the lower-left position.
    """

    ## TODO: check syntax for indexing and accessing elements of a matrix!!
    # upper-left
    img_ul = visual.ImageStim(win = window,
                              image = all_images[stim_position[trial][0]],
                              units = "norm",
                              pos = (-0.45, 0.45)
                        )
    # upper-right
    img_ur = visual.ImageStim(win = window,
                              image = all_images[stim_position[trial][1]],
                              units = "norm",
                              pos = (0.45, 0.45)
                        )
    # lower-right
    img_lr = visual.ImageStim(win = window,
                              image = all_images[stim_position[trial][2]],
                              units = "norm",
                              pos = (0.45, -0.45)
                        )
    # lower-left
    img_ll = visual.ImageStim(win = window,
                              image = all_images[stim_position[trial][3]],
                              units = "norm",
                              pos = (-0.45, -0.45)
                        )

    return img_ul, img_ur, img_lr, img_ll


#-------------------------------------------------------------------------------
## TODO: check syntax for indexing and accessing elements of a matrix!!
def get_img_chosen_by_subj(resp, all_images, stim_position, trial):
    """
    Returns the name of the image file for the stimulus selected by the
    subject's response.
    """

    if resp == "f":
        img_chosen = all_images[stim_position[trial][0]]
    elif resp == "j":
        img_chosen = all_images[stim_position[trial][1]]
    elif resp == "h":
        img_chosen = all_images[stim_position[trial][2]]
    elif resp == "g":
        img_chosen = all_images[stim_position[trial][3]]
    else:
        img_chosen = "error"

    return img_chosen


#-------------------------------------------------------------------------------
def get_number_of_img_chosen_by_subj(resp, all_images, stim_position,
                                     payoff, trial):
    """
    Returns the number of the image selected by the subject's response.
    """

    if resp == "f":
        num_of_img_chosen = int(all_images[stim_position[trial][0]][9]) -1
    elif resp == "j":
        num_of_img_chosen = int(all_images[stim_position[trial][1]][9]) -1
    elif resp == "h":
        num_of_img_chosen = int(all_images[stim_position[trial][2]][9]) -1
    elif resp == "g":
        num_of_img_chosen = int(all_images[stim_position[trial][3]][9]) -1
    else:
        num_of_img_chosen = 999

    return num_of_img_chosen


#-------------------------------------------------------------------------------
def get_payoff_matrix(ntrials = 192):
    """
    Returns the payoffs matrix for the four-armed bandit task as implemented by
    Daw, O'Doherty, Dayan, Seymour & Dolan (2006). The payoff matrix has
    dimensions equal to ntrials x 4 (each column corresponds to one of the
    four images).
    There are four slot machines. The reward for each one is a random value
    from a Gaussian distribution. The mean of the distribution is a function of
    the mean determined on the previous trial; the standard deviation is kept
    constant. In this manner, the payoff of the four slot machines varies over
    time, by following a random walk. The mu of the Gaussian distributions in
    each trial are saved in a matrix mu, with 4 columns, one for each slot
    machine, and ntrial rows. The payoffs for each trial and each slot machine
    are thus saved in a (ntrials x 4) matrix.
    The payoff for choosing the ith slot machine on trial t was
    between 1 and 100 points, drawn from a Gaussian distribution
    (standard deviation σo = 4) around a mean μi,t and rounded to the
    nearest integer. At each timestep, the means diffused in a
    decaying Gaussian random walk, with
    μ_[i, t+1] = λμ_[i, t] + (1 - λ)θ + ν for each i. The decay parameter
    λ was 0.9836, the decay center θ was 50, and the diffusion noise
    ν was zero-mean Gaussian (standard deviation σd = 2.8)
    See Supplementary Material, first page.
    For each bandit, there are *independent* random walks specifying the
    probability of a gain (0: no, +1: yes) and of a loss (0: no; -1: yes)
    on each trial -- see Seymour et al. (2006) fig. 1b.
    This means that, on each trial, there are three possible outcomes:
    +1, 0, -1. Graphically, they can be represented by the euro image (gain),
    the crossed euro (loss), and an uniforme gray patch for nothing.
    Therefore, the feedback screen can have two images: the gain (present
    or absent) on the left, and the loss (present or absent) on the right.
    The random walks across different bandits are also independent.
    """
    # Parameters definitions for the random walk.
    # Parameters definitions for the random walk.
    lmbda = 0.9836
    theta = 50
    # ntrials = 300
    mu_start_1 = 80
    mu_start_2 = 60
    mu_start_3 = 40
    mu_start_4 = 20

    mu = np.zeros((ntrials, 4))
    payoff = np.zeros((ntrials, 4))

    mu[0, 0] = mu_start_1
    mu[0, 1] = mu_start_2
    mu[0, 2] = mu_start_3
    mu[0, 3] = mu_start_4

    stay_in_loop = 1

    while stay_in_loop:
        for cond in range(4):
            for i in range(1, ntrials):
                mu[i, cond] = lmbda * mu[i - 1, cond] + (1 - lmbda) * theta + np.random.normal(0, 2.8, 1)
            for i in range(0, ntrials):
                payoff[i, cond] = np.random.normal(mu[i, cond], 4, 1)
        if np.std(payoff) > 19 and np.min(payoff) > 0 and np.max(payoff) < 100:
            stay_in_loop = 0

    return payoff


#-------------------------------------------------------------------------------
def get_reward_prob_for_chosen_image(resp_key, all_images, stim_position,
                                     payoff, trial):
    """
    Returns the reward probability for the image chosen by the subject.
    """

    # "img_chosen" is the name of the file of the image chosen by the subject.
    img_chosen = get_img_chosen_by_subj(
        resp_key, all_images, stim_position, trial
    )

    # "num_img_chosen" is the image number (0, 1, 2, or 3) chosen by the
    # subject.
    num_img_chosen = get_number_of_img_chosen_by_subj(
        resp_key, all_images, stim_position, payoff, trial
    )

    # Get the trial-th row of the stim_position matrix. This 4-element array
    # is such that the zero-th element refers to the image that is shown in the
    # upper-left position, the first-st element refers to the image that is
    # shown in the upper-right position, and so on. The values in the four
    # elements vector are related to the name of the image: 0: img_a, 1:
    # img_b, etc. Example: [2 1 0 3] means that img_c (i.e., 2) is located in
    # the upper-left position, etc.
    row_arr = stim_position[trial, :]

    # "row_arr" is of class numpy.ndarray and must be converted to a list.
    row_list = row_arr.tolist()

    # Sanity check.
    # dtype_before = type(row_arr)
    # dtype_after = type(row_list)
    # print("Data type before converting = {}\nData type after converting = {}".format(dtype_before, dtype_after))

    # Get the position of "num_img_chosen" within the list "row_list".
    index_of_chosen_img_within_row = row_list.index(num_img_chosen)

    # Sanity check.
    # print(row_list)
    # print(num_img_chosen)
    # print(index_of_chosen_img_within_row)
    # print(payoff[trial])
    # print(payoff[trial][index_of_chosen_img_within_row])

    # Probability of positive feedback for chosen image.
    payoff_for_chosen_img = payoff[trial][index_of_chosen_img_within_row] / 100

    return payoff_for_chosen_img


#-------------------------------------------------------------------------------
def show_fixation(window, fixation):
    # Draw fixation cross.
    fixation.draw()
    window.flip()
    core.wait(1)
