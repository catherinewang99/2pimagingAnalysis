# 2pimagingAnalysis
Repository to store code I write to analyze my two-photon calcium imaging data

## class Session (session.py)
Session stores most of the core pre-processing and basic analysis functions to look at a single session's neural data combined with behavioral data.
Accompanied by selectivityAnalysis python file.

### Functions
| Pre-processing functions        ||
| ----------- | ----------- |
| determine_cutoff      | Finds the shortest trial length out of all trials    |
| find_low_mean_F   | Finds and crop low F trials that correspond to water leaks        |
|get_pearsonscorr_neuron | Filters neurons based on the consistency of their signal|
| plot_mean_F | Plots mean F for all neurons over trials in session |
|crop_trials | Removes trials from i_good_trials based on inputs |
| normalize_all_by_neural_baseline | Normalize all neurons by each neuron's trial-averaged F0 | 
|normalize_all_by_baseline |  Normalize all neurons by each neuron's F0 on each trial |
| normalize_by_histogram | Normalize all neurons by each neuron's F0 based on bottom quantile over all trials |
|normalize_all_by_histogram | Normalize all neurons by each neuron's F0 based on bottom quantile for each trial |
|normalize_z_score  | Z-score normalizes all neurons traces |


| Data organization functions  ||
| ------------ | ---------------|
| lick_correct_direction | Finds trial numbers corresponding to correct lick in specified direction |
| lick_incorrect_direction | Finds trial numbers corresponding to incorrect lick in specified direction |
| lick_actual_direction | Finds trial numbers corresponding to an actual lick direction |
| get_trace_matrix | Returns matrices of dF/F0 traces over right/left trials of a single neuron |
| get_trace_matrix_multiple |Returns matrices of dF/F0 traces averaged over right/left trials of multiple neurons |


| Plotting functions  ||
| ------------ | ---------------|
| plot_PSTH | Plots single neuron PSTH over R/L trials |
| plot_single_trial_PSTH | Plots single neuron PSTH on a single trial |
| plot_population_PSTH | Plots many neurons PSTH over R/L trials |
| plot_selectivity | Plots a single line representing selectivity of given neuron over trial |
| plot_contra_ipsi_pop | Plots contra and ipsi preferring neurons' traces in two plots |
| plot_prefer_nonprefer | Plots preferred and nonpreferred traces for all selective neurons in one graph |
| plot_individual_raster | Plots greyscale heatmap-style graph of a single neuron across all trials |
| plot_left_right_raster | Plots greyscale heatmap-style graph of a single neuron right trials then left trials |
| plot_raster_and_PSTH | Plot heatmap then averaged L/R trace for a single neuron | 
| plot_rasterPSTH_sidebyside |  Plot heatmap then averaged L/R trace for a single neuron comparing control and opto trials |
| plot_number_of_sig_neurons | Plots number of contra / ipsi neurons over course of trial | 
| selectivity_table_by_epoch | Plots table of L/R traces of selective neurons over three epochs and contra/ipsi population proportions | 
| plot_three_selectivity | Plots selectivity traces over three epochs and number of neurons in each population |
| selectivity_optogenetics | Plots overall selectivity trace across opto vs control trials |
| single_neuron_sel | Plots proportion of stim/lick/reward/mixed cells over trial using two different methods| 
| stim_choice_outcome_selectivity | Plots selectivity traces of stim/lick/reward/mixed cells using Susu's method | 





| Selectivity functions  ||
| ------------ | ---------------|
| get_epoch_selective | Identifies neurons that are selective in a given epoch |
| screen_preference | Determine if a neuron is left or right preferring |
| contra_ipsi_pop | Finds neurons that are left and right preferring |

| Behavioral state functions || 
| --------------------- | --------|
| find_bias_trials | Finds trials belonging to behavioral states calculated via the GLM-HMM or other method |
| plot_prefer_nonprefer_sidebyside | Plots preferred and nonpreferred traces for all selective neurons in control vs bias trials |
| plot_pref_overstates | Plots preferred and nonpreferred traces for all selective neurons across 3 behavioral states and control |
| plot_selectivity_overstates | Plots selectivity traces for all selective neurons across 3 behavioral states and control in one graph |



Unused/broken functions:
- crop_baseline
- reject_outliers
- normalize_by_baseline
- filter_by_deltas
- population_sel_timecourse

## class QC (quality.py)

Accompanied by qualityAnalysis.py script.

## class Mode (activityMode.py)

activityMode has functions to look at population activity in terms of the various activity modes. 
Accompanied by activity_mode_analysis.py script.

### Functions

## class Sample (bootstrap.py)

Class used to sample neurons, trials, and sessions to do decoding analysis. Uses method outlined in Susu et al. 2023, bioArxiv. Accompanied by notebook. 

## class behavior
