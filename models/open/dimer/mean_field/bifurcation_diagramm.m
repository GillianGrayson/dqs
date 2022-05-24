clear all;

path = 'E:/YandexDisk/Work/qs/models/open/dimer/mean_field';

U_start = 0.005;
U_shift = 0.005;
U_num = 100;
A = 3.4 ;
J = 1.0;

omega = 1.0;
phase = 0.0;
gamma = 0.1;

npt = 100;
np = 5000;

N = 1000;

states = linspace(1, N, N);

Us = zeros(U_num, 1);
BD = zeros(U_num , N);

seed_num = 1;

num_lpns = 2;

lpn_exps = zeros(U_num, num_lpns);

for U_id = 1 : U_num
    
    U = U_start + U_shift * (U_id - 1)
    Us(U_id) = U;
    
    for seed = 1:seed_num
        
        fn_suffix = sprintf('params(%0.4f_%0.4f_%0.4f)_mod(%0.4f_%0.4f_%0.4f)', ...
            J, ...
            U, ...
            gamma, ...
            A, ...
            omega, ...
            phase);
        
        fn = sprintf('%s/data/%s/data.txt', path, fn_suffix);
        data = importdata(fn);
        nu = data(2:end,1);
        coordinate = N/2*(cos(nu)+1);
        for per_id = 1 : np
            tmp = coordinate(per_id) / N * (N-1);
            id = floor(tmp) + 1;
            BD(U_id, id) = BD(U_id, id) + 1;
        end
        
        fn = sprintf('%s/data/%s/lyaps.txt', path, fn_suffix);
        data_lyaps = importdata(fn);
        for lpn_id = 1:num_lpns
            lpn_exps(U_id, lpn_id) = lpn_exps(U_id, lpn_id) + mean(data_lyaps(:, lpn_id)) / seed_num;
        end

    end
    
    BD(U_id, :) = BD(U_id, :) / max(BD(U_id, :));
    
end

fig = figure;
propertyeditor(fig);

subplot(2,1,1);
hLine = imagesc(Us, states, BD');
set(gca, 'FontSize', 24);
xlabel('$U$', 'Interpreter', 'latex');
ylabel('$n$', 'Interpreter', 'latex');
set(gca,'xticklabel',{[]})
xlim([Us(1), Us(end)])
colormap hot;
h = colorbar;
title(h, '$\mathrm{PDF}$', 'Interpreter', 'latex');
set(gca,'YDir','normal');
set(gca, 'Position', [0.13 0.5 0.77 0.41]);
hold all;

subplot(2,1,2);
for lpn_id = 1:num_lpns
    h = plot(Us, lpn_exps(:, lpn_id), 'LineWidth', 2);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    hold all;
end
h = plot([Us(1), Us(end)], [0, 0],'black', 'LineStyle', ':', 'LineWidth', 2);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
set(gca, 'Position', [0.13 0.13 0.77 0.35]);
set(gca, 'FontSize', 24);
xlim([Us(1), Us(end)])
xlabel('$U$', 'Interpreter', 'latex');
ylabel('$\lambda$', 'Interpreter', 'latex');

fn_suffix = sprintf('params(%0.4f_%0.4f)_mod(%0.4f_%0.4f_%0.4f)', ...
    J, ...
    gamma, ...
    A, ...
    omega, ...
    phase);
oqs_save_fig(fig, sprintf('%s/figures/from_U_%s.fig', path, fn_suffix))