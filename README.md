# 2pimagingAnalysis
Repository to store code I write to analyze my two-photon calcium imaging data

## Functions in class Session

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


| Selectivity functions  ||
| ------------ | ---------------|
| get_epoch_selective | Identifies neurons that are selective in a given epoch |
| screen_preference | Determine if a neuron is left or right preferring |
| contra_ipsi_pop | 




Unused/broken functions:
- crop_baseline
- reject_outliers
- normalize_by_baseline
