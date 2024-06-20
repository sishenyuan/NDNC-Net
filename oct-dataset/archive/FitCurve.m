%% Run fourierseriesfit.m in batch scale
clear;

load_dir = 'waveform_data';
save_dir = 'waveform_data_fitted';

files = dir(fullfile(load_dir, '*.csv'));
for i = 1:length(files)
    fileName = files(i).name;
    disp(['Processing ', fileName, ', Progress: ', num2str(i), '/', num2str(length(files))]);
    fourierseriesfit(fullfile(load_dir, fileName), save_dir, 20);
end