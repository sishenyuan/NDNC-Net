%% Generate dynamic curve data by varying friction and IPM's initial position
friction_array = [0.12];
ipm_init_pos_array = [45e-3];

mdl = 'Monomagnet_random';
disp(['Loading model: ', mdl, '...']);
load_system(mdl)

friction_path = [mdl '/mu'];
ipm_init_pos_path = [mdl '/ipm_init_pos'];
save_dir = 'dynamics_data';
image_dir = 'dynamics_images';

if ~exist(save_dir, 'dir')
    disp(['Creating directory: ', save_dir, '...']);
    mkdir(save_dir);
end

if ~exist(image_dir, 'dir')
    disp(['Creating directory: ', image_dir, '...']);
    mkdir(image_dir);
end

disp('Start generating dynamic curve data...');
for i = 1:length(friction_array)
    for j = 1:length(ipm_init_pos_array)
        disp(['Progress: ', num2str((i-1)*length(ipm_init_pos_array) + j), '/', num2str(length(friction_array)*length(ipm_init_pos_array))]);
        set_param(friction_path, 'Value', num2str(friction_array(i)));
        set_param(ipm_init_pos_path, 'Value', num2str(ipm_init_pos_array(j)));
        simout = sim(mdl);
        
        %% Save results
        time = simout.RPS.time;
        rps = simout.RPS.signals.values;
        filename = fullfile(save_dir, ['dynamics_', num2str(friction_array(i)), '_', num2str(ipm_init_pos_array(j)*10e3), '.csv']);
        data = [time, rps];
        writematrix(data, filename);

        %% Plot results
        figure;
        plot(time, rps);
        title(['Friction: ', num2str(friction_array(i)), ', IPM init pos: ', num2str(ipm_init_pos_array(j)*10e3)]);
        xlabel('Time (s)');
        ylabel('RPS');
        
        clear simout time rps data;
    end
end
