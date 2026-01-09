clearvars;
plotSettings;

data_path = '/Volumes/ppc_2/crissp/perrotol/Martin_CSL/graph_controllabillity';
text_file = fullfile(data_path,"csv/save_NEB_txt/val_phon_mean_distrib_calib_FSE_test_by_layer.txt");
fid = fopen(text_file, 'r');
P = textscan(fid, "%s %s %s %s %s", 'Delimiter','|');
fclose(fid);

file_calib_control = fullfile(data_path,"_NEB_Fastspeech2_120000_phonPred_predictor_embeddings_test_control/layer_6");
file_ref = fullfile(data_path,"_NEB_Fastspeech2_120000_phonPred_predictor_embeddings_test_emb_by_layer_mean_distrib_calib");
% Mean pitch values
mean_pitch = 91.053;
std_pitch = 3.57;

% Mean energy values
mean_energy = 65.386;
std_energy = 8.36;

index_utt = 3;
phon_utt = P{3}{index_utt};
text_utt = P{4}{index_utt};
split_phon = phon_input_to_cell_array(phon_utt);

vuv_phon = ones(size(split_phon));
vuv_phon = vuv_phon ...
    - strcmp(split_phon,'p') ...
    - strcmp(split_phon,'t') ...
    - strcmp(split_phon,'k') ...
    - strcmp(split_phon,'f') ...
    - strcmp(split_phon,'s') ...
    - strcmp(split_phon,' ') ...
    - strcmp(split_phon,',');

list_bias_f0 = [
    516;
    0;
    -516;
];
nbr_bias = length(list_bias_f0);
% colors_by_style = distinguishable_colors(nbr_bias);
colors_by_style = [pcol.gipsaGAIA ; pcol.gipsaPSD ; pcol.gipsaPPC];

% Get duration
load(sprintf("%s/TEST%05d_syn_log_duration", file_ref, index_utt)); % log_duration_mat
log_duration_mat_unbiased = double(log_duration_mat);
duration_mat_unbiased = (exp(log_duration_mat_unbiased)-1)*256/22050;
t_lim_unbiased = cumsum(duration_mat_unbiased);
t_centre_unbiased = t_lim_unbiased-duration_mat_unbiased/2;
t_lim_unbiased = [0 t_lim_unbiased];
[x,fs] = audioread(sprintf("%s/TEST%05d_syn.wav", file_ref, index_utt));

%% Display norm pdf


% PDF
x = (mean_pitch-20):0.01:(mean_pitch+20);
y = normpdf(x,mean_pitch,std_pitch);

figure(10);
clf
plot(x,y)
line([1 1]*(mean_pitch-2*std_pitch),[0 max(y)*1.1],'Color','k')
line([1 1]*(mean_pitch-std_pitch),[0 max(y)*1.1],'Color','k')
line([1 1]*(mean_pitch+std_pitch),[0 max(y)*1.1],'Color','k')
line([1 1]*(mean_pitch+2*std_pitch),[0 max(y)*1.1],'Color','k')
ylim([0 max(y)*1.1])
grid on




%%
figure(1)
fp = get(gcf,'Position');
fp(3:4) = [1500 600];
set(gcf,'Position',fp)
set(gcf,'Renderer', 'painters');
set(gcf,'PaperSize',[53 21])
clf;

posPlot = plotPosition(2,1,[0.04 0.1 0.0 0.01],[0.01 0.01 0.07 0.09]);





ptxt.title.size = 20;
ptxt.text.size = 18;
ptxt.label.size = 20;
ptxt.legend.size = 20;

f0_lim = [83 101];
% f0_lim = [85 101];
% E_lim = [0 0.13];
E_lim = [-4 5];
% E_lim = [48 80];

subplotMargin(posPlot,1);
% grid on;
% title(sprintf('Text: "%s"', text_utt));
title('Pitch prediction with f_0 control','Interpreter','tex');
% xticks(1:utt_length);
% xticks(t_lim_unbiased);
% xticks(t_centre_unbiased);
% xticklabels(split_phon);
% xtickangle(0);
% xlim([0, utt_length+1]);
xlim([0 t_lim_unbiased(end)]);
ylim(f0_lim);
yyaxis right
line([0,t_lim_unbiased(end)],[1 1]*(mean_pitch+2*std_pitch),'LineStyle', '--', 'color', colors_by_style(1, :), 'LineWidth',1, 'HandleVisibility', 'off'); hold on
line([0,t_lim_unbiased(end)],[1 1]*(mean_pitch+std_pitch),'LineStyle', '--', 'color', colors_by_style(1, :), 'LineWidth',1, 'HandleVisibility', 'off'); hold on
line([0,t_lim_unbiased(end)],[1 1]*(mean_pitch),'LineStyle', '--', 'color', colors_by_style(2, :), 'LineWidth',1, 'HandleVisibility', 'off'); hold on
line([0,t_lim_unbiased(end)],[1 1]*(mean_pitch-std_pitch),'LineStyle', '--', 'color', colors_by_style(3, :), 'LineWidth',1, 'HandleVisibility', 'off'); hold on
line([0,t_lim_unbiased(end)],[1 1]*(mean_pitch-2*std_pitch),'LineStyle', '--', 'color', colors_by_style(3, :), 'LineWidth',1, 'HandleVisibility', 'off'); hold on
set(gca,'Ytick',mean_pitch+std_pitch*(-2:2),'YtickLabel',{'-2 std','-1 std','0 std','+1 std','+2 std'},'YColor','k')
ylabel({'ground truth f_o','distribution'},'Interpreter','tex');
ylim(f0_lim);
yyaxis left
for i = 1:length(t_lim_unbiased)-1
    if ~vuv_phon(i)
        patch([t_lim_unbiased(i) t_lim_unbiased(i+1) t_lim_unbiased(i+1) t_lim_unbiased(i)], [f0_lim(1) f0_lim(1) f0_lim(2) f0_lim(2)],[1 1 1]*0.6,'EdgeColor','none','FaceAlpha',0.1, 'HandleVisibility', 'off'); hold on
    end
    line([1 1]*t_lim_unbiased(i+1),f0_lim,'Color',[1 1 1]*0.6, 'HandleVisibility', 'off');
    text(t_centre_unbiased(i),diff(f0_lim)*0.95+f0_lim(1),split_phon{i},'HorizontalAlignment','center');
end
% xlabel('Phonetic Inputs');
xlabel('Time (s)');
ylabel('f_o (semitones)','Interpreter','tex');
ax = gca;
ax.YGrid = 'on';
box on;



subplotMargin(posPlot,2);
grid on;
% title(sprintf('Text: "%s"', text_utt),'Fontsize',sizeTitle);
title('Energy covariation with f_o control','Interpreter','tex');
% xticks(1:utt_length);
% xticks(t_lim_unbiased);
% xticklabels(split_phon);
% xtickangle(0);
% xlim([0, utt_length+1]);
% xlabel('Phonetic Inputs');
set(gca,'Ytick',-4:2:4)
xlim([0 t_lim_unbiased(end)]);
ylim(E_lim);
for i = 1:length(t_lim_unbiased)-1
    if ~vuv_phon(i)
        patch([t_lim_unbiased(i) t_lim_unbiased(i+1) t_lim_unbiased(i+1) t_lim_unbiased(i)], [E_lim(1) E_lim(1) E_lim(2) E_lim(2)],[1 1 1]*0.6,'EdgeColor','none','FaceAlpha',0.1, 'HandleVisibility', 'off'); hold on
    end
    line([1 1]*t_lim_unbiased(i+1),E_lim,'Color',[1 1 1]*0.6, 'HandleVisibility', 'off');
    text(t_centre_unbiased(i),diff(E_lim)*0.95+E_lim(1),split_phon{i},'HorizontalAlignment','center');
end
xlabel('Time (s)');
% ylabel('Signal amplitude (linear scale)');
ylabel('\Delta Energy (dB)');
%legend({'+2 std'; 'Unbiased'; '-2 std'});
box on;



for i_bias = [2 1 3]%1:nbr_bias
    current_bias = list_bias_f0(i_bias);
    if current_bias < 0
        label_bias = 'moins';
    else
        label_bias = 'plus';
    end
    absolute_value = abs(current_bias);
    if current_bias == 0
        load(sprintf("%s/TEST%05d_syn_pitch", file_ref, index_utt)); % pitch_prediction_mat
        load(sprintf("%s/TEST%05d_syn_energy", file_ref, index_utt)); % energy_prediction_mat
    else
        sub_folder = sprintf("pitch_control_bias_%s_%d", label_bias, absolute_value);
        load(sprintf("%s/%s/TEST%05d_syn_pitch", file_calib_control, sub_folder, index_utt)); 
        load(sprintf("%s/%s/TEST%05d_syn_energy", file_calib_control, sub_folder, index_utt)); 
        load(sprintf("%s/%s/TEST%05d_syn_log_duration", file_calib_control, sub_folder, index_utt)); 

    end
    
    utt_length = length(pitch_prediction_mat);
    pitch_prediction_mat = std_pitch*pitch_prediction_mat + mean_pitch;
    energy_prediction_mat = std_energy*energy_prediction_mat + mean_energy;
%     energy_prediction_mat = dbSPL2mag(std_energy*energy_prediction_mat + mean_energy);
    log_duration_mat_biased = log_duration_mat;
    duration_mat_biased = (exp(log_duration_mat_biased)-1)*256/22050;
    
    if i_bias == 2
        energy_prediction_mat_unbiased = energy_prediction_mat;
    end
    energy_prediction_mat = energy_prediction_mat-energy_prediction_mat_unbiased;
    energy_prediction_mat(~vuv_phon) = nan;
    
    % Premier subplot pour la pitch prediction
    subplotMargin(posPlot,1);

    % plot(1:utt_length, pitch_prediction_mat, 'color', colors_by_style(i_bias, :), 'LineWidth',3);
    plot(t_centre_unbiased, pitch_prediction_mat,'.-', 'color', colors_by_style(i_bias, :), 'LineWidth',3,'MarkerSize',21);
    hold on;
    % plot([0,t_lim_unbiased(end)], mean(pitch_prediction_mat)*ones(1,2), '--', 'color', colors_by_style(i_bias, :), 'LineWidth',3, 'HandleVisibility', 'off');

    % Second subplot pour l'energy prediction
    subplotMargin(posPlot,2);
    
    % plot(1:utt_length, energy_prediction_mat, 'color', colors_by_style(i_bias, :), 'LineWidth',3);
    plot(t_centre_unbiased, energy_prediction_mat,'.-', 'color', colors_by_style(i_bias, :), 'LineWidth',3,'MarkerSize',21);
    hold on;
    % plot([0,t_lim_unbiased(end)], mean(energy_prediction_mat)*ones(1,2), '--', 'color', colors_by_style(i_bias, :), 'LineWidth',3, 'HandleVisibility', 'off');
end

lgnd = legend({'unbiased';'+2 std';'-2 std'});
lgnd.Position(1) = 0.92;
lgnd.Position(2) = 0.42;
title(lgnd,'f_o control')

setPlotFonts(ptxt);
% saveas(gcf,cfg.results_root,'pdf');
