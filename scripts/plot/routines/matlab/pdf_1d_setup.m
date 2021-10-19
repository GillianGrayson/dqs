function pdf = pdf_1d_setup(pdf)

pdf.x_bin_shift = (pdf.x_bin_f - pdf.x_bin_s) / pdf.x_num_bins;

pdf.x_bin_centers = linspace(...
    pdf.x_bin_s + 0.5 * pdf.x_bin_shift, ...
    pdf.x_bin_f - 0.5 * pdf.x_bin_shift, ...
    pdf.x_num_bins)';

pdf.pdf = zeros(pdf.x_num_bins, 1);
pdf.inc_count = 0;
pdf.not_inc_count = 0;

end
