# 2pimagingAnalysis
Repository to store code I write to analyze my two-photon calcium imaging data

## Functions in session.py

| Pre-processing functions        ||
| ----------- | ----------- |
| determine_cutoff      | Finds the shortest trial length out of all trials    |
| find_low_mean_F   | Finds and crop low F trials that correspond to water leaks        |
|get_pearsonscorr_neuron | Filters neurons based on the consistency of their signal|
| plot_mean_F | Plots mean F for all neurons over trials in session |
|crop_trials | Removes trials from i_good_trials based on inputs |


| Data organization functions  ||
| ------------ | ---------------|
| lick_correct_direction | Finds trial numbers corresponding to correct lick in specified direction |
| lick_incorrect_direction | Finds trial numbers corresponding to incorrect lick in specified direction |
| lick_actual_direction | Finds trial numbers corresponding to an actual lick direction |
| get_trace_matrix | Returns matrices of dF/F0 traces over right/left trials of a single neuron |
| get_trace_matrix_multiple |Returns matrices of dF/F0 traces averaged over right/left trials of multiple neurons |






Unused/broken functions:
- crop_baseline
- reject_outliers
