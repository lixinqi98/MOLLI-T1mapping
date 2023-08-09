%% add path
addpath("functions");
warning('off')
pwd_path = pwd;
%% MOLLI fitting



name = '/Users/mona/Library/CloudStorage/Box-Box/Animals/Lisbon/Lisbon_wk8+2/PostconT1w/POSTCON1_T1MAP_SAX5_0127/LISBON_WEEK8+2_07JAN2022.MR.HEART_CEDARS.0127.0001.2022.01.07.13.00.16.274088.31690714.IMA';

x = dicomread(name);

processed_x = x;
processed_x(processed_x > 2000) = 0;
processed_x = (processed_x - min(processed_x(:))) / (max(processed_x(:)) - min(processed_x(:)));

contour_file = '/Users/mona/Library/CloudStorage/Box-Box/Animals/Lisbon/Lisbon_wk8+2/registration/endo_epi.nii';
contour = niftiread(contour_file)';

[x_1, y_1] = size(x);
epi_BW = contour;
epi_BW(contour==2)=0;

boundary_epi = boundarymask(epi_BW);

endo_BW = contour;
endo_BW(contour==1)=0;

boundary_endo = boundarymask(endo_BW);
boundary = boundary_endo + boundary_epi;
% figure('Position', [1, 1, 1100, 100])
z_1 = 1;
t = tiledlayout(1,z_1);
for i=1:z_1
    orig_slice = x;
    ax1 = nexttile; axis off,imshow(labeloverlay(mat2gray(orig_slice),boundary,'Transparency',0)) 
end
t.TileSpacing = 'tight';
t.Padding = 'tight';

saveas(gcf,sprintf("/Users/mona/Library/CloudStorage/Box-Box/Animals/Lisbon/Lisbon_wk8+2/registration/endo_epi_contour.png"));

% saveas(gcf,sprintf("%s/MOLLI_%s_%d_regi_vols.png", label, subjectid, slice));
