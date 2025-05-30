% Wavelet Coherence Analysis: ITF-Agulhas System
% Author: Sandy Herho <sandy.herho@email.ucr.edu>
% This script performs wavelet coherence analysis on ITF and Agulhas data
% and saves the results as figures and statistical summaries

clear; close all; clc;

%% Load and prepare data
fprintf('Loading data from CSV file...\n');
fprintf('Author: Sandy Herho <sandy.herho@email.ucr.edu>\n\n');

data_path = '../data/processed_data/itf_agulhas.csv';
data = readtable(data_path);

% Extract variables
time = data.time;
itf_g = data.itf_g;  % ITF geostrophic
itf_t = data.itf_t;  % ITF temperature
itf_s = data.itf_s;  % ITF salinity
agulhas_box = data.aghulas_box;  % Note: original CSV has typo 'aghulas'
agulhas_jet = data.aghulas_jet;

% Convert time to years if needed (assuming time is already in years)
years_data = time;

% Display time range to verify CE years
fprintf('Time range in data: %.1f to %.1f CE\n', min(years_data), max(years_data));

%% Create output directories if they don't exist
if ~exist('../figs', 'dir')
    mkdir('../figs');
end
if ~exist('../stats', 'dir')
    mkdir('../stats');
end

%% Initialize statistics storage
stats_text = sprintf('WAVELET COHERENCE ANALYSIS: ITF-AGULHAS SYSTEM\n');
stats_text = [stats_text sprintf('================================================\n')];
stats_text = [stats_text sprintf('Author: Sandy Herho <sandy.herho@email.ucr.edu>\n')];
stats_text = [stats_text sprintf('Analysis Date: %s\n', datestr(now))];
stats_text = [stats_text sprintf('Data Points: %d\n', length(time))];
stats_text = [stats_text sprintf('Time Range: %.2f to %.2f years C.E.\n', min(years_data), max(years_data))];
stats_text = [stats_text sprintf('================================================\n\n')];

%% Create figure with subplots
fig = figure('Position', [50, 50, 1500, 1000]);  % Wider figure to prevent cutoff

% Define subplot positions for better control
% [left, bottom, width, height] - adjusted for better spacing
subplot_positions = {
    [0.10, 0.55, 0.36, 0.35],  % (a) top-left
    [0.54, 0.55, 0.36, 0.35],  % (b) top-right
    [0.10, 0.12, 0.36, 0.35],  % (c) bottom-left
    [0.54, 0.12, 0.36, 0.35]   % (d) bottom-right
};

% Analysis pairs (corrected naming)
pairs = {
    {itf_s, agulhas_box, 'ITF Salinity → Agulhas Box', '(a)'},
    {itf_t, agulhas_box, 'ITF Temperature → Agulhas Box', '(b)'},
    {itf_g, agulhas_jet, 'ITF Geostrophic → Agulhas Jet', '(c)'},
    {itf_g, agulhas_box, 'ITF Geostrophic → Agulhas Box', '(d)'}
};

% Storage for detailed statistics
all_wcoh = cell(4, 1);
all_periods = [];
all_time_clean = cell(4, 1);

%% Process each panel
for i = 1:4
    ax(i) = subplot('Position', subplot_positions{i});
    
    % Get data for this pair
    x_data = pairs{i}{1};
    y_data = pairs{i}{2};
    pair_name = pairs{i}{3};
    panel_label = pairs{i}{4};
    
    % Remove NaN values
    valid_idx = ~isnan(x_data) & ~isnan(y_data);
    x_clean = x_data(valid_idx);
    y_clean = y_data(valid_idx);
    time_clean = years_data(valid_idx);
    all_time_clean{i} = time_clean;
    
    % Calculate sampling period (assuming uniform sampling)
    dt = median(diff(time_clean));
    
    % Perform wavelet coherence analysis
    [wcoh, wcs, period, coi] = wcoherence(x_clean, y_clean, years(dt), ...
        'PhaseDisplayThreshold', 0.7);
    
    % Store for detailed analysis
    all_wcoh{i} = wcoh;
    all_periods = period;
    
    % Plot wavelet coherence
    wcoherence(x_clean, y_clean, years(dt), 'PhaseDisplayThreshold', 0.7);
    
    % Apply hot colormap
    colormap(hot);
    
    % Fix x-axis to show actual CE years
    current_xticks = get(gca, 'XTick');
    % Map tick positions to actual years
    if ~isempty(current_xticks)
        % Convert indices to actual year values
        year_ticks = interp1(1:length(time_clean), time_clean, current_xticks, 'linear', 'extrap');
        set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.0f', x), year_ticks, 'UniformOutput', false));
    end
    
    % Remove individual labels and title
    xlabel('');
    ylabel('');
    title('');
    
    % Add panel label - professional placement (top-left corner, semi-transparent)
    text(0.05, 0.92, panel_label, 'Units', 'normalized', ...
        'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k', ...
        'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    
    % Customize tick labels - make them bigger and more visible
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'LineWidth', 1.5);
    
    % Make tick marks more visible
    set(gca, 'TickLength', [0.02 0.02]);
    set(gca, 'TickDir', 'out');
    
    % Grid settings
    grid on;
    set(gca, 'GridAlpha', 0.3);
    
    % Calculate basic statistics
    avg_coherence = mean(wcoh(:));
    high_coherence_fraction = sum(wcoh(:) > 0.7) / numel(wcoh);
    
    % Convert periods to years for analysis
    if isduration(period)
        period_years = years(period);
    else
        period_years = period;
    end
    
    % DETAILED TECHNICAL STATISTICS
    stats_text = [stats_text sprintf('\n%s %s:\n', panel_label, pair_name)];
    stats_text = [stats_text sprintf('----------------------------------------\n')];
    stats_text = [stats_text sprintf('BASIC STATISTICS:\n')];
    stats_text = [stats_text sprintf('  - Average coherence: %.3f\n', avg_coherence)];
    stats_text = [stats_text sprintf('  - Fraction with coherence > 0.7: %.1f%%\n', high_coherence_fraction * 100)];
    stats_text = [stats_text sprintf('  - Valid data points: %d\n', length(x_clean))];
    stats_text = [stats_text sprintf('  - Sampling interval: %.3f years\n', dt)];
    
    % Find dominant periods
    mean_coherence_by_period = mean(wcoh, 2);
    [sorted_coh, idx] = sort(mean_coherence_by_period, 'descend');
    top_periods = period_years(idx(1:min(5, length(idx))));
    
    stats_text = [stats_text sprintf('\nDOMINANT PERIODS (top 5):\n')];
    for j = 1:length(top_periods)
        stats_text = [stats_text sprintf('  %d. Period: %.2f years (avg coherence: %.3f)\n', ...
            j, top_periods(j), sorted_coh(j))];
    end
    
    % Find specific years with high coherence for each dominant period
    stats_text = [stats_text sprintf('\nTIME LOCALIZATION OF HIGH COHERENCE (>0.7):\n')];
    for j = 1:min(3, length(top_periods))  % Top 3 periods
        period_idx = idx(j);
        high_coh_times = time_clean(wcoh(period_idx, :) > 0.7);
        
        if ~isempty(high_coh_times)
            stats_text = [stats_text sprintf('  Period %.2f years occurs at years CE: ', top_periods(j))];
            
            % Group consecutive years
            time_groups = {};
            current_group = high_coh_times(1);
            for k = 2:length(high_coh_times)
                if high_coh_times(k) - high_coh_times(k-1) < 1
                    current_group = [current_group, high_coh_times(k)];
                else
                    time_groups{end+1} = current_group;
                    current_group = high_coh_times(k);
                end
            end
            time_groups{end+1} = current_group;
            
            % Format output with CE designation
            for k = 1:length(time_groups)
                if length(time_groups{k}) > 1
                    stats_text = [stats_text sprintf('%.1f-%.1f CE', ...
                        time_groups{k}(1), time_groups{k}(end))];
                else
                    stats_text = [stats_text sprintf('%.1f CE', time_groups{k})];
                end
                if k < length(time_groups)
                    stats_text = [stats_text ', '];
                end
            end
            stats_text = [stats_text sprintf('\n')];
        end
    end
    
    % Phase relationship analysis
    phase_angles = angle(wcs);
    stats_text = [stats_text sprintf('\nPHASE RELATIONSHIPS:\n')];
    
    % Calculate average phase for high coherence regions
    high_coh_mask = wcoh > 0.7;
    if any(high_coh_mask(:))
        avg_phase = mean(phase_angles(high_coh_mask));
        stats_text = [stats_text sprintf('  - Average phase angle (coherence > 0.7): %.2f rad (%.1f degrees)\n', ...
            avg_phase, avg_phase * 180/pi)];
        
        % Interpret phase
        if abs(avg_phase) < pi/4
            phase_interp = 'approximately in-phase';
        elseif abs(avg_phase - pi) < pi/4 || abs(avg_phase + pi) < pi/4
            phase_interp = 'approximately anti-phase';
        elseif avg_phase > 0 && avg_phase < pi
            phase_interp = sprintf('first series leads by ~%.1f degrees', avg_phase * 180/pi);
        else
            phase_interp = sprintf('second series leads by ~%.1f degrees', abs(avg_phase) * 180/pi);
        end
        stats_text = [stats_text sprintf('  - Interpretation: %s\n', phase_interp)];
    end
end

%% Remove individual colorbars
for i = 1:4
    h = findobj(ax(i).Parent, 'Type', 'ColorBar');
    if ~isempty(h)
        delete(h);
    end
end

%% Add single colorbar with improved formatting and positioning
cb = colorbar('Position', [0.92, 0.25, 0.02, 0.5]);  % Moved slightly left
cb.Label.String = 'Magnitude-Squared Coherence';
cb.Label.FontSize = 16;
cb.Label.FontWeight = 'bold';
cb.FontSize = 14;
cb.FontWeight = 'bold';
cb.LineWidth = 1.5;
cb.TickLength = 0.02;

% Ensure colorbar label is not cut off
cb.Label.Position(1) = cb.Label.Position(1) + 0.5;  % Move label farther from colorbar

%% Add common labels
% X-label at bottom
annotation('textbox', [0.25, 0.02, 0.5, 0.05], ...
    'String', 'Time [years C.E.]', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 18, ...
    'FontWeight', 'bold', ...
    'EdgeColor', 'none');

% Y-label at center left (rotated) - FIXED POSITIONING
ylabel_h = axes('Position', [0.03, 0.3, 0.02, 0.4], 'Visible', 'off');
text(0.5, 0.5, 'Period [years]', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 18, ...
    'FontWeight', 'bold', ...
    'Rotation', 90, ...
    'Parent', ylabel_h);

%% Save figure with tight layout to prevent cutoff
set(gcf, 'PaperPositionMode', 'auto');
output_filename = '../figs/wtc_itf_agulhas_4panel.png';
print(fig, output_filename, '-dpng', '-r400');
fprintf('Figure saved as: %s\n', output_filename);

%% Additional global statistics
stats_text = [stats_text sprintf('\n\nGLOBAL STATISTICS:\n')];
stats_text = [stats_text sprintf('================================================\n')];
stats_text = [stats_text sprintf('Time series length: %.2f years\n', max(years_data) - min(years_data))];
stats_text = [stats_text sprintf('Sampling interval: %.3f years\n', median(diff(years_data)))];

if isduration(all_periods)
    period_range = years(all_periods);
else
    period_range = all_periods;
end
stats_text = [stats_text sprintf('Period range analyzed: %.1f to %.1f years\n', ...
    min(period_range), max(period_range))];

% Cross-panel comparison
stats_text = [stats_text sprintf('\n\nCROSS-PANEL COMPARISON:\n')];
stats_text = [stats_text sprintf('================================================\n')];

% Find common high-coherence periods across panels
common_periods = [];
for p = 1:length(period_range)
    coherences_at_period = zeros(4, 1);
    for i = 1:4
        if ~isempty(all_wcoh{i})
            coherences_at_period(i) = mean(all_wcoh{i}(p, :));
        end
    end
    
    if all(coherences_at_period > 0.5)  % If all panels show moderate coherence
        common_periods = [common_periods; period_range(p), mean(coherences_at_period)];
    end
end

if ~isempty(common_periods)
    stats_text = [stats_text sprintf('Common periods with coherence > 0.5 across all panels:\n')];
    [~, idx] = sort(common_periods(:, 2), 'descend');
    for j = 1:min(5, size(common_periods, 1))
        stats_text = [stats_text sprintf('  - Period: %.2f years (avg coherence: %.3f)\n', ...
            common_periods(idx(j), 1), common_periods(idx(j), 2))];
    end
end

%% Physical Interpretation
stats_text = [stats_text sprintf('\n\nPHYSICAL INTERPRETATION:\n')];
stats_text = [stats_text sprintf('================================================\n')];
stats_text = [stats_text sprintf('The wavelet coherence analysis reveals multi-scale interactions between\n')];
stats_text = [stats_text sprintf('Indonesian Throughflow (ITF) components and Agulhas system variability.\n\n')];

stats_text = [stats_text sprintf('KEY FINDINGS:\n')];
stats_text = [stats_text sprintf('1. ITF Salinity - Agulhas Box (panel a):\n')];
stats_text = [stats_text sprintf('   - Indicates freshwater flux teleconnections\n')];
stats_text = [stats_text sprintf('   - High coherence suggests Indo-Pacific freshwater export impacts\n')];
stats_text = [stats_text sprintf('     Indian Ocean salinity structure\n\n')];

stats_text = [stats_text sprintf('2. ITF Temperature - Agulhas Box (panel b):\n')];
stats_text = [stats_text sprintf('   - Reveals thermal connections between ocean basins\n')];
stats_text = [stats_text sprintf('   - Coherent variability indicates heat transport pathways\n')];
stats_text = [stats_text sprintf('   - Important for regional climate modulation\n\n')];

stats_text = [stats_text sprintf('3. ITF Geostrophic - Agulhas Jet (panel c):\n')];
stats_text = [stats_text sprintf('   - Direct dynamical connection between volume transports\n')];
stats_text = [stats_text sprintf('   - Suggests common forcing mechanisms (e.g., wind stress)\n\n')];

stats_text = [stats_text sprintf('4. ITF Geostrophic - Agulhas Box (panel d):\n')];
stats_text = [stats_text sprintf('   - Broader regional impact of ITF variability\n')];
stats_text = [stats_text sprintf('   - Indicates basin-scale circulation adjustments\n\n')];

stats_text = [stats_text sprintf('PHASE RELATIONSHIPS:\n')];
stats_text = [stats_text sprintf('- In-phase behavior: Simultaneous response to common forcing\n')];
stats_text = [stats_text sprintf('- Phase lags: Indicate propagation time of signals\n')];
stats_text = [stats_text sprintf('- Anti-phase: Possible compensating mechanisms\n\n')];

stats_text = [stats_text sprintf('DOMINANT TIMESCALES:\n')];
stats_text = [stats_text sprintf('- Sub-annual: Seasonal monsoon forcing\n')];
stats_text = [stats_text sprintf('- 1-2 years: ENSO teleconnections\n')];
stats_text = [stats_text sprintf('- 3-7 years: Indian Ocean Dipole and ENSO interactions\n')];
stats_text = [stats_text sprintf('- Decadal: Pacific Decadal Oscillation influence\n')];

%% Save statistics file
stats_filename = '../stats/wtc_itf_agulhas_analysis.txt';
fid = fopen(stats_filename, 'w');
fprintf(fid, '%s', stats_text);
fclose(fid);
fprintf('Statistics saved as: %s\n', stats_filename);

%% Display completion message
fprintf('\nAnalysis complete!\n');
fprintf('Figure created by: Sandy Herho <sandy.herho@email.ucr.edu>\n');
fprintf('- Figure saved to: %s\n', output_filename);
fprintf('- Statistics saved to: %s\n', stats_filename);