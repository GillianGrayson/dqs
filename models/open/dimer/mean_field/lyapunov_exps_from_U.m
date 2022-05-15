clear all;

data_path = 'E:/YandexDisk/Work/qs/data/models/open/dimer/mean_field';

U_start = 0.01;
U_shift = 0.01;
U_num = 50;
A = 3.4 ;
J = 1.0;

omega = 1.0;
phase = 0.0;
gamma = 0.1;

npt = 100;
np = 1000;

N = 1000;

seed_num = 1;

num_lpns = 2;

lpn_exps = zeros(U_num, num_lpns);

Us = zeros(U_num, 1);

for U_id = 1 : U_num
    
    U = U_start + U_shift * (U_id - 1)
    Us(U_id) = U;
    
    for seed = 1:seed_num
        
       fn_suffix = sprintf('params(%0.4f_%0.4f_%0.4f)_mod(%0.4f_%0.4f_%0.4f)/lyaps.txt', ...
            J, ...
            U, ...
            gamma, ...
            A, ...
            omega, ...
            phase);
        
        fn = sprintf('%s/%s', data_path, fn_suffix);
        data = importdata(fn);
        
        for lpn_id = 1:num_lpns
            lpn_exps(U_id, lpn_id) = lpn_exps(U_id, lpn_id) + mean(data(:, lpn_id)) / seed_num;
        end
        
    end
    
end


fig = figure;

propertyeditor(fig);

for lpn_id = 1:num_lpns
    hLine = plot(Us, lpn_exps(:, lpn_id), 'LineWidth', 2);
    legend(hLine, sprintf('\lambda #%d', lpn_id));
    hold all;
end

set(gca, 'FontSize', 30);
xlabel('$U$', 'Interpreter', 'latex');
