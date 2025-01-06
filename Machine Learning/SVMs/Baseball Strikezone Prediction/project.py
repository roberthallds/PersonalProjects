# ------------------------------------------
# BASEBALL STRIKE ZONE PREDICTION USING
# SUPPORT VECTOR MACHINES
#
# A project for my Codecademy Certified
# Data Scientist: Machine Learning Specialist
# professional certification.
# 
# Data from pybaseball on Aaron Judge was used
# for this project. Data was aggregated from 
# the pybaseball library by Codecademy. 
# 
# Robert Hall
# 01/06/2025
# ------------------------------------------

import matplotlib.pyplot as plt # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from svm_visualization import draw_boundary
from players import aaron_judge

fig, ax = plt.subplots()

# 1. Get Columns for Aaron Judge
print(aaron_judge.columns)

'''
Index(['pitch_type', 'game_date', 'release_speed', 'release_pos_x',
       'release_pos_z', 'player_name', 'batter', 'pitcher', 'events',
       'description', 'spin_dir', 'spin_rate_deprecated',
       'break_angle_deprecated', 'break_length_deprecated', 'zone', 'des',
       'game_type', 'stand', 'p_throws', 'home_team', 'away_team', 'type',
       'hit_location', 'bb_type', 'balls', 'strikes', 'game_year', 'pfx_x',
       'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b',
       'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y',
       'tfs_deprecated', 'tfs_zulu_deprecated', 'pos2_person_id', 'umpire',
       'sv_id', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot',
       'hit_distance_sc', 'launch_speed', 'launch_angle', 'effective_speed',
       'release_spin_rate', 'release_extension', 'game_pk', 'pos1_person_id',
       'pos2_person_id.1', 'pos3_person_id', 'pos4_person_id',
       'pos5_person_id', 'pos6_person_id', 'pos7_person_id', 'pos8_person_id',
       'pos9_person_id', 'release_pos_y', 'estimated_ba_using_speedangle',
       'estimated_woba_using_speedangle', 'woba_value', 'woba_denom',
       'babip_value', 'iso_value', 'launch_speed_angle', 'at_bat_number',
       'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score',
       'fld_score', 'post_away_score', 'post_home_score', 'post_bat_score',
       'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment'],
      dtype='object')
'''

# 2. Get Description Feature Unique Values
print(aaron_judge.description.unique())

'''
['swinging_strike' 'called_strike' 'ball' 'hit_into_play_score' 'foul'
 'blocked_ball' 'hit_into_play' 'hit_into_play_no_out'
 'swinging_strike_blocked' 'foul_tip' 'hit_by_pitch']
'''

# 3. Get Judge Ball/Strike Information 
print(aaron_judge.type.unique()) 
# 'S' for Strike
# 'B' for Ball
# 'X' for Hit or Out

'''
['S' 'B' 'X']
'''

# 4. Define Labels for SVM training
aaron_judge['type_binary'] = aaron_judge['type'].map({"S": 1, "B": 0})
# Strike = 1
# Ball = 0

# 5. Examine unique strike type values 
#    post-transformation
print(aaron_judge.type_binary.unique()) # check

'''
[ 1.  0. nan]
'''

# 6. Examine left-right pitch distance from 
#    home plate
print(aaron_judge['plate_x'].head())

'''
0    1.0150
1    0.4546
2    0.0957
3    1.5161
4    0.0764
Name: plate_x, dtype: float64
'''

# 7. Remove null values in distance features
#    and effective null values (hits & balls)
#    in the strike 'type' feature
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'type_binary'])
print(aaron_judge.type_binary.unique()) #check

'''
[1. 0.]
'''

# 8. Create scatterplot to visualize
#    horizontal distance against vertical
#    distance
plt.scatter(x=aaron_judge['plate_x'], y=aaron_judge['plate_z'], c=aaron_judge['type_binary'], cmap=plt.cm.coolwarm, alpha=0.25)
plt.title("Strikezone Vertical Distance Against \nHorizontal Distance \n(Model with Default Gamma and C)")
plt.xlabel("Horizontal Distance")
plt.ylabel("Vertical Distance")

# 9. Split dataset into training and test data
training_set, validation_set = train_test_split(aaron_judge, test_size=0.2, random_state = 47)

# 10. Create Support Vector Classifier (SVC)
classifier = SVC(kernel='rbf')

# 11. Fit model to training data
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type_binary'])

# 12. Visualize SVM on Cartesian Plane
draw_boundary(ax, classifier)
plt.show()
plt.savefig('default_parameters.png', format='png')
plt.clf()

# 13. Score the model
print(f"Model Score: {classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type_binary'])}")

'''
Model Score: 0.8566037735849057
'''

# 14-15. Tune Hyperparameters
best_params = {'value': 0, 'gamma': 1, 'C': 1} 
# default scores at each incremental gamma and c value
for gamma in range(1, 5):
  for c in range(1, 5):
    classifier = SVC(kernel='rbf', gamma=gamma, C=c)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type_binary'])
    score_j = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type_binary'])
    if (score_j > best_params['value']):
      best_params['value'] = score_j
      best_params['gamma'] = gamma
      best_params['C'] = c
print(f"Best Score is {best_params['value']} with Gamma {best_params['gamma']} and C {best_params['C']}. ")

'''
Best Score is 0.8566037735849057 with Gamma 3 and C 1. 
'''

# 16-17. Refit new classifier with optimal hyperparameters and visualize
best_classifier = SVC(kernel='rbf', gamma=3, C=1)
best_classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type_binary'])
plt.scatter(x=aaron_judge['plate_x'], y=aaron_judge['plate_z'], c=aaron_judge['type_binary'], cmap=plt.cm.coolwarm, alpha=0.25)
plt.title("Strikezone Vertical Distance Against \nHorizontal Distance")
plt.xlabel("Horizontal Distance")
plt.ylabel("Vertical Distance")
draw_boundary(ax, best_classifier)
plt.show()
plt.clf()