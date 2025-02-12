% Define vmin and vmax values for both groups of neurons
vmin = min([expert_response_stack(:); expert_stack(:); naive_response_stack(:); naive_stack(:)]);
vmax = max([expert_response_stack(:); expert_stack(:); naive_response_stack(:); naive_stack(:)]);
%% 
% Load or define your data matrices (e.g., right_stack, left_stack, right_stack_post, left_stack_post)
vmin = min([expert_stack(:); naive_stack(:)])+3.7;
vmax = max([expert_stack(:); naive_stack(:)])-1.5;

% Set the 'parula' colormap
colormap(parula);

% Create a figure for the first heatmap
f1 = figure;
ax1 = axes;
right_im = imagesc(expert_stack, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');
title('Expert delay neurons');
xline(7, '--w');
xline(26, '--w');
xline(71, '--w');
% Create a figure for the second heatmap
f2 = figure;
ax2 = axes;
leftim = imagesc(naive_stack, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');
title('Naive delay neurons');
xline(7, '--w');
xline(26, '--w');
xline(71, '--w');
% Create a figure for the third heatmap
% f3 = figure;
% ax3 = axes;
% right_im_post = imagesc(right_stack_post, [vmin, vmax]);
% axis off;
% % colorbar('Location', 'southoutside');
% title('Session 2 - Right Trials');
% xline(9, '--w');
% xline(17, '--w');
% xline(35, '--w');
% % Create a figure for the fourth heatmap
% f4 = figure;
% ax4 = axes;
% leftim_post = imagesc(left_stack_post, [vmin, vmax]);
% axis off;
% % colorbar('Location', 'southoutside');
% title('Session 2 - Left Trials');
% xline(9, '--w');
% xline(17, '--w');
% xline(35, '--w');

% Save the figures as needed
print(f1, 'H:\data\BAYLORCW044\python\expert_delay_neurons.pdf', '-dpdf', '-bestfit');
print(f2, 'H:\data\BAYLORCW044\python\naive_delay_neurons.pdf', '-dpdf', '-bestfit');
% print(f3, 'F:\data\SFN 2023\nsession2_right_trials.pdf', '-dpdf', '-bestfit');
% print(f4, 'F:\data\SFN 2023\nsession2_left_trials.pdf', '-dpdf', '-bestfit');

%% response neurons
% Define vmin and vmax values
vmin = min([expert_response_stack(:); naive_response_stack(:)])+2;
vmax = max([expert_response_stack(:); naive_response_stack(:)])-1.5;


% Load or define your data matrices (e.g., right_stack, left_stack, right_stack_post, left_stack_post)

% Set the 'parula' colormap
colormap(parula);

% Create a figure for the first heatmap
f1 = figure;
ax1 = axes;
right_im = imagesc(expert_response_stack, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');
title('Expert response neurons');
xline(7, '--w');
xline(26, '--w');
xline(71, '--w');
% Create a figure for the second heatmap
f2 = figure;
ax2 = axes;
leftim = imagesc(naive_response_stack, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');
title('Naive response neurons');
xline(7, '--w');
xline(26, '--w');
xline(71, '--w');
% Create a figure for the third heatmap
% f3 = figure;
% ax3 = axes;
% right_im_post = imagesc(right_stack_post, [vmin, vmax]);
% axis off;
% % colorbar('Location', 'southoutside');
% title('Session 2 - Right Trials');
% xline(9, '--w');
% xline(17, '--w');
% xline(35, '--w');
% % Create a figure for the fourth heatmap
% f4 = figure;
% ax4 = axes;
% leftim_post = imagesc(left_stack_post, [vmin, vmax]);
% axis off;
% % colorbar('Location', 'southoutside');
% title('Session 2 - Left Trials');
% xline(9, '--w');
% xline(17, '--w');
% xline(35, '--w');

% Save the figures as needed
print(f1, 'H:\data\BAYLORCW044\python\expert_response_neurons.pdf', '-dpdf', '-bestfit');
print(f2, 'H:\data\BAYLORCW044\python\naive_response_neurons.pdf', '-dpdf', '-bestfit');
% print(f3, 'F:\data\SFN 2023\nsession2_right_trials.pdf', '-dpdf', '-bestfit');
% print(f4, 'F:\data\SFN 2023\nsession2_left_trials.pdf', '-dpdf', '-bestfit');



%% 


% Create a 2x2 subplot figure
f = figure;
axarr = tight_subplot(2, 2, 0.01, 0.1, 0.1);

% Define vmin and vmax values
vmin = 0;
vmax = 0.3;

% Load or define your data matrices (e.g., right_stack, left_stack, right_stack_post, left_stack_post)

% Set the 'parula' colormap
colormap(parula);

% FIRST SESS
% Right trials first
% right_stack = normalize(right_stack(:, 7:end));
% right_stack = right_stack(2:end, :);
axes(axarr(3)); % MATLAB uses 1-based indexing
right_im = imagesc(right_stack, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');
title('Session 1');
ylabel('Right trials');

% Left trials
% left_stack = normalize(left_stack(:, 7:end));
% left_stack = left_stack(2:end, :);
axes(axarr(1));
leftim = imagesc(left_stack, [vmin, vmax]);
axis off;
ylabel('Left trials');

% SECOND SESS
% Right trials first
% right_stack_post = normalize(right_stack_post(:, 7:end));
% right_stack_post = right_stack_post(2:end, :);
axes(axarr(4));
right_im = imagesc(right_stack_post, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');
title('Session 2');

% Left trials
% left_stack_post = normalize(left_stack_post(:, 7:end));
% left_stack_post = left_stack_post(2:end, :);
axes(axarr(2));
leftim = imagesc(left_stack_post, [vmin, vmax]);
axis off;
% colorbar('Location', 'southoutside');

% Save the figure
print('F:\data\SFN 2023\trained_matched_pop_matlab.pdf', '-dpdf', '-bestfit');

% Define the tight_subplot function to create subplots with the specified spacing
function ha = tight_subplot(Nh, Nw, gap, marg_h, marg_w)
if nargin < 4
    marg_h = 0.01;
end
if nargin < 5
    marg_w = 0.01;
end
pos = get(gcf, 'Position');
un = get(gcf, 'Units');
set(gcf, 'Units', 'pixels');
ppos = get(gcf, 'Position');
set(gcf, 'Units', un);
pw = ppos(3);
ph = ppos(4);

axh = (1 - marg_h - (Nh - 1) * gap) / Nh;
axw = (1 - marg_w - (Nw - 1) * gap) / Nw;

ii = 0;
for i = 1:Nh
    for j = 1:Nw
        ii = ii + 1;
        ha(ii) = axes('Units', 'normalized', 'Position', ...
            [marg_w + (j - 1) * (axw + gap), 1 - i * (axh + gap) + marg_h, axw, axh]);
    end
end
end


