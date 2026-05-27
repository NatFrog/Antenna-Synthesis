%% Ideal 4x4 MATLAB patterns paired with datasets_4x4consistent_hfss
%
% Reads (dphase_x_deg, dphase_y_deg) from each consistent HFSS CSV and runs
% simulate_array.m (same physics as ArrayPattern_4x4_dataset.m / setup.m).
%
% Prerequisites:
%   - setup.m on path (M=N=4, GainPlot_1x1.csv or grid CSV)
%   - datasets_4x4consistent_hfss/patterns_global_*.csv
%
% Usage (from repo root):
%   derive_4x4_matlab_from_consistent_hfss
%   derive_4x4_matlab_from_consistent_hfss(true)   % smoke: first file only

function derive_4x4_matlab_from_consistent_hfss(smokeTest)
    if nargin < 1, smokeTest = false; end

    setup;  % loads elem_gain_lin, TH, PH, M, N, dx, dy, k, normalize_to_0dB

    hfssDir = fullfile('datasets_4x4consistent_hfss');
    outDir  = fullfile('datasets_4x4consistent_matlab', 'datasets_4x4');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    listing = dir(fullfile(hfssDir, 'patterns_global_*.csv'));
    listing = sortrows(listing, 'name');
    if smokeTest
        listing = listing(1);
    end

    fprintf('Writing %d file(s) -> %s\n', numel(listing), outDir);

    for f = 1:numel(listing)
        inPath = fullfile(hfssDir, listing(f).name);
        T = readtable(inPath, 'VariableNamingRule', 'preserve');

        sampleCols = T.Properties.VariableNames(3:end);
        nS = numel(sampleCols);
        m = regexp(listing(f).name, 'patterns_global_(\d+)', 'tokens');
        fileIdx = str2double(m{1}{1});
        startIdx = (fileIdx - 1) * nS + 1;

        dphase_x = zeros(1, nS);
        dphase_y = zeros(1, nS);
        phi_peak = zeros(1, nS);
        theta_peak = zeros(1, nS);

        theta_flat = T.theta_deg(5:end);
        phi_flat   = T.phi_deg(5:end);
        numAngles  = numel(theta_flat);
        pattern_all = zeros(numAngles, nS);

        for s = 1:nS
            col = sampleCols{s};
            dphase_x(s) = T.(col)(1);
            dphase_y(s) = T.(col)(2);

            G_dB = simulate_array(M, N, dphase_x(s), dphase_y(s), ...
                elem_gain_lin, TH, PH, dx, dy, k, normalize_to_0dB);
            pattern_all(:, s) = G_dB(:);

            [~, idx] = max(G_dB(:));
            [ith, iph] = ind2sub(size(G_dB), idx);
            theta_peak(s) = rad2deg(TH(ith, iph));
            phi_peak(s)   = rad2deg(PH(ith, iph));
        end

        headers = [{'theta_deg', 'phi_deg'}, ...
            arrayfun(@(i) sprintf('s%05d', startIdx + i - 1), 1:nS, 'UniformOutput', false)];

        dx_row = [{'dphase_x_deg', ''}, num2cell(dphase_x)];
        dy_row = [{'dphase_y_deg', ''}, num2cell(dphase_y)];
        ph_row = [{'phi_peak_deg', ''}, num2cell(phi_peak)];
        th_row = [{'theta_peak_deg', ''}, num2cell(theta_peak)];
        data_rows = [num2cell(theta_flat), num2cell(phi_flat), num2cell(pattern_all)];

        C = [headers; dx_row; dy_row; ph_row; th_row; data_rows];
        outPath = fullfile(outDir, listing(f).name);
        writecell(C, outPath);

        fprintf('  [%3d/%3d] %s (%d samples)\n', f, numel(listing), listing(f).name, nS);
    end

    fprintf('Done.\n');
end
