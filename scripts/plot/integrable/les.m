clear all;
addpath('../routines/matlab')

N = 7;
tau = 1;
k = -1;
n_seeds = 1000;

lpn_type = 0;
lpn_log_deltas = [-6.0]';
Ts = [5]';

is_scaled = 0;

path = 'E:/YandexDisk/Work/os_lnd/draft/mbl/2/figures/integrable/lambda/';

for log_delta_id = 1:size(lpn_log_deltas, 1)
    fig = figure;
    
    if is_scaled == 1
        x = [-5:.01:5];
        y = normpdf(x,0,1);
        h = plot(x, y, 'LineWidth', 5);
        legend(h, sprintf('Normal Distribution'));
        hold all;
    end
    
    for T_id = 1:size(Ts, 1)
        fn = sprintf('%s/lambda_N(%d)_numSeeds(%d)_tau(%d)_k(%d)_T(%0.4f)_lpn(%d_%0.4f).csv', ...
            path, ...
            N, ...
            n_seeds, ...
            tau, ...
            k, ...
            Ts(T_id), ...
            lpn_type, ...
            lpn_log_deltas(log_delta_id));
        
        data = importdata(fn);
        lambdas = data(:);
        
        if is_scaled == 1
            mean_lambdas = mean(lambdas);
            std_lambdas = std(lambdas);
            lambdas = (lambdas - mean_lambdas) / std(lambdas);
        end
        
        [h, p, jbstat, critval] = jbtest(lambdas)
        [h, p, kstat, critval] = lillietest(lambdas)
        [h, p, adstat, cv] = adtest(lambdas)
        
        pdf.x_num_bins = 100;
        pdf.x_label = '$\lambda$';
        if is_scaled == 1
            pdf.x_label = '$\bar{\lambda}$';
        end
        min_x = min(lambdas);
        max_x = max(lambdas);
        pdf.x_bin_s = -max(abs(min_x), abs(max_x));
        pdf.x_bin_f = -pdf.x_bin_s;
        pdf = pdf_1d_setup(pdf);
        pdf = pdf_1d_update(pdf, lambdas);
        pdf = pdf_1d_release(pdf);
        h = plot(pdf.x_bin_centers, pdf.pdf, 'LineWidth', 2);
        legend(h, sprintf('$\\tau=%d$', Ts(T_id)))
        legend(gca,'off');
        legend('Interpreter', 'latex');
        set(gca, 'FontSize', 30);
        xlabel(pdf.x_label, 'Interpreter', 'latex');
        set(gca, 'FontSize', 30);
        ylabel('$PDF$', 'Interpreter', 'latex');
        hold all;
    end
    
    fn_fig = sprintf('%s/density_lambda_N(%d)_numSeeds(%d)_tau(%d)_k(%d)_lpn(%d_%0.4f)', ...
        path, ...
        N, ...
        n_seeds, ...
        tau, ...
        k, ...
        lpn_type, ...
        lpn_log_deltas(log_delta_id));
    if is_scaled == 1
        fn_fig = sprintf('%s_scaled', fn_fig);
    end
    save_fig(fig, fn_fig);
end
