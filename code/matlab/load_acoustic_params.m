%% load_acoustic_params.m
% Returns the acoustic parameters per character computed with Praat.
%
function acoustic_params = load_acoustic_params(ind_utt, path)
% Global parameters
ref_level = 20e-6; % ref_audio in pascal for sound pressure in db (SPL)
ref_semiton = 1; % ref 1 Hz
min_freq = 0;
max_freq = 8000;
nbr_filters_mel = 80;
fs_frame = 22050/256;
freq_unit = 'semitone'; % Hz, mel, mel_index, semitone
% freq_unit = 'Hz'; % Hz, mel, mel_index, semitone
freq_interpolation = false;
list_pct = {' ','?','!',':',';','.','§','~','[',']','(',')','-','"','¬', ',', '«', '»', '#', '__', '!_'};
list_vowels = {'a'; 'u'; 'i'; 'e'; 'e^'; 'y'; 'x'; 'x^'; 'o'; 'o^'; 'x~'; 'e~'; 'o~'; 'a~';};


% Load acoustic params
% Load acoustic params
if ~exist(sprintf('%sTEST%05d_acoustic_params.csv',path,ind_utt))
    acoustic_params = nan(1, 33);
    return
end
P = read_csv(sprintf('%sTEST%05d_acoustic_params.csv',path,ind_utt),31);
len_utt = length(P{1});
list_char = P{1};
list_start = P{2};
list_end = P{3};
list_gt_duration = P{4};
list_z_score_gt = P{5};
list_z_score_align = P{6};
list_mean_f0_utt = P{7};
list_std_f0_utt = P{8};
list_f0_inter1 = P{9};
list_f0_inter2 = P{10};
list_f0_inter3 = P{11};
list_f1_inter1 = P{12};
list_f1_inter2 = P{13};
list_f1_inter3 = P{14};
list_f2_onset = P{15};
list_f2_inter1 = P{16};
list_f2_inter2 = P{17};
list_f2_inter3 = P{18};
list_f3_inter1 = P{19};
list_f3_inter2 = P{20};
list_f3_inter3 = P{21};
list_rms = P{22};
list_center_gravity = P{23};
list_std_gravity = P{24};
list_absolute_pos = P{25};
list_absolute_pos_end = P{26};
list_relative_pos = P{27};
list_relative_pos_end = P{28};
list_rms_utt = P{29};
list_hnr = P{30};
list_spectral_balance = P{31};
nbr_pitch_cwt_coef = 0;
acoustic_params = zeros(len_utt, 35+nbr_pitch_cwt_coef);

% Duration
% acoustic_params(:,1) = log(1000*(str2double(list_end) - str2double(list_start)));
% acoustic_params(:,1) = log(1 + round((str2double(list_end) - str2double(list_start))*fs_frame));
acoustic_params(:,1) = log(1 + (str2double(list_end) - str2double(list_start))*fs_frame);

% Ground-Truth duration
acoustic_params(:,2) = 1000*str2double(list_gt_duration);

% Z-Score GT
acoustic_params(:,3) = str2double(list_z_score_gt);
% Z-score activation duration
acoustic_params(:,4) = str2double(list_z_score_align);

if freq_interpolation
    list_phon = 1:len_utt;
    list_f0_inter2 = str2double(list_f0_inter2);
    if sum(~isnan(list_f0_inter2)) > 1
        list_f0_inter2 = interp1(list_phon(~isnan(list_f0_inter2)), list_f0_inter2(~isnan(list_f0_inter2)), list_phon, 'linear', 'extrap');
    end
end

% F0 pitch, F1, F2 and F3 (first third of duration)
if strcmp(freq_unit,'Hz')
    % Mean F0 Utt
    acoustic_params(:,5) = str2double(list_mean_f0_utt);
    % std F0 Utt
    acoustic_params(:,6) = str2double(list_std_f0_utt);

    acoustic_params(:,7) = str2double(list_f0_inter1);
    acoustic_params(:,8) = str2double(list_f0_inter2);
    acoustic_params(:,9) = str2double(list_f0_inter3);
    
    acoustic_params(:,10) = str2double(list_f1_inter1);
    acoustic_params(:,11) = str2double(list_f1_inter2);
    acoustic_params(:,12) = str2double(list_f1_inter3);
    
    acoustic_params(:,13) = str2double(list_f2_onset);
    acoustic_params(:,14) = str2double(list_f2_inter1);
    acoustic_params(:,15) = str2double(list_f2_inter2);
    acoustic_params(:,16) = str2double(list_f2_inter3);
    
    acoustic_params(:,17) = str2double(list_f3_inter1);
    acoustic_params(:,18) = str2double(list_f3_inter2);
    acoustic_params(:,19) = str2double(list_f3_inter3);
elseif strcmp(freq_unit,'mel')
    % Mean F0 Utt
    acoustic_params(:,5) = linear_freq_to_mel(str2double(list_mean_f0_utt)');
    % std F0 Utt
    acoustic_params(:,6) = linear_freq_to_mel(str2double(list_std_f0_utt)');
    
    acoustic_params(:,7) = linear_freq_to_mel(str2double(list_f0_inter1)');
    acoustic_params(:,8) = linear_freq_to_mel(str2double(list_f0_inter2)');
    acoustic_params(:,9) = linear_freq_to_mel(str2double(list_f0_inter3)');
    
    acoustic_params(:,10) = linear_freq_to_mel(str2double(list_f1_inter1)');
    acoustic_params(:,11) = linear_freq_to_mel(str2double(list_f1_inter2)');
    acoustic_params(:,12) = linear_freq_to_mel(str2double(list_f1_inter3)');
    
    acoustic_params(:,13) = linear_freq_to_mel(str2double(list_f2_onset)');
    acoustic_params(:,14) = linear_freq_to_mel(str2double(list_f2_inter1)');
    acoustic_params(:,15) = linear_freq_to_mel(str2double(list_f2_inter2)');
    acoustic_params(:,16) = linear_freq_to_mel(str2double(list_f2_inter3)');
    
    acoustic_params(:,17) = linear_freq_to_mel(str2double(list_f3_inter1)');
    acoustic_params(:,18) = linear_freq_to_mel(str2double(list_f3_inter2)');
    acoustic_params(:,19) = linear_freq_to_mel(str2double(list_f3_inter3)');
elseif strcmp(freq_unit,'mel_index')
    % Mean F0 Utt
    acoustic_params(:,5) = compute_mel_filter_index(str2double(list_mean_f0_utt)', min_freq, max_freq, nbr_filters_mel);
    % std F0 Utt
    acoustic_params(:,6) = compute_mel_filter_index(str2double(list_std_f0_utt)', min_freq, max_freq, nbr_filters_mel);
    
    acoustic_params(:,7) = compute_mel_filter_index(str2double(list_f0_inter1)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,8) = compute_mel_filter_index(str2double(list_f0_inter2)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,9) = compute_mel_filter_index(str2double(list_f0_inter3)', min_freq, max_freq, nbr_filters_mel);
    
    acoustic_params(:,10) = compute_mel_filter_index(str2double(list_f1_inter1)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,11) = compute_mel_filter_index(str2double(list_f1_inter2)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,12) = compute_mel_filter_index(str2double(list_f1_inter3)', min_freq, max_freq, nbr_filters_mel);
    
    acoustic_params(:,13) = compute_mel_filter_index(str2double(list_f2_onset)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,14) = compute_mel_filter_index(str2double(list_f2_inter1)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,15) = compute_mel_filter_index(str2double(list_f2_inter2)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,16) = compute_mel_filter_index(str2double(list_f2_inter3)', min_freq, max_freq, nbr_filters_mel);
    
    acoustic_params(:,17) = compute_mel_filter_index(str2double(list_f3_inter1)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,18) = compute_mel_filter_index(str2double(list_f3_inter2)', min_freq, max_freq, nbr_filters_mel);
    acoustic_params(:,19) = compute_mel_filter_index(str2double(list_f3_inter3)', min_freq, max_freq, nbr_filters_mel);
elseif strcmp(freq_unit,'semitone')
    % Mean F0 Utt
    acoustic_params(:,5) = linear_freq_to_semitone(str2double(list_mean_f0_utt)', ref_semiton);
    % std F0 Utt
    acoustic_params(:,6) = linear_freq_to_semitone(str2double(list_std_f0_utt)', ref_semiton);
    
    acoustic_params(:,7) = linear_freq_to_semitone(str2double(list_f0_inter1)', ref_semiton);
%     acoustic_params(:,8) = linear_freq_to_semitone(list_f0_inter2, ref_semiton);
    acoustic_params(:,8) = linear_freq_to_semitone(str2double(list_f0_inter2)', ref_semiton);
    acoustic_params(:,9) = linear_freq_to_semitone(str2double(list_f0_inter3)', ref_semiton);
    
    acoustic_params(:,10) = linear_freq_to_semitone(str2double(list_f1_inter1)', ref_semiton);
    acoustic_params(:,11) = linear_freq_to_semitone(str2double(list_f1_inter2)', ref_semiton);
    acoustic_params(:,12) = linear_freq_to_semitone(str2double(list_f1_inter3)', ref_semiton);
    
    acoustic_params(:,13) = linear_freq_to_semitone(str2double(list_f2_onset)', ref_semiton);
    acoustic_params(:,14) = linear_freq_to_semitone(str2double(list_f2_inter1)', ref_semiton);
    acoustic_params(:,15) = linear_freq_to_semitone(str2double(list_f2_inter2)', ref_semiton);
    acoustic_params(:,16) = linear_freq_to_semitone(str2double(list_f2_inter3)', ref_semiton);
    
    acoustic_params(:,17) = linear_freq_to_semitone(str2double(list_f3_inter1)', ref_semiton);
    acoustic_params(:,18) = linear_freq_to_semitone(str2double(list_f3_inter2)', ref_semiton);
    acoustic_params(:,19) = linear_freq_to_semitone(str2double(list_f3_inter3)', ref_semiton);
end

% Spectral slope (db/decade)
all_slope = calculate_spectral_slope_by_char(ind_utt, path);
acoustic_params(:,20) = all_slope;

% Sound Pressure (in db (SPL))
all_rms = 20*log10(str2double(list_rms)/ref_level);
acoustic_params(:,21) = all_rms;

% Center of gravity of spectrum
acoustic_params(:,22) = linear_freq_to_semitone(str2double(list_center_gravity)', ref_semiton);
% Standard deviation of center of gravity of spectrum
acoustic_params(:,23) = str2double(list_std_gravity);

% Absolute position in utterance
acoustic_params(:,24) = str2double(list_absolute_pos);
% Absolute position from end in utterance
acoustic_params(:,25) = str2double(list_absolute_pos_end);
% Relative position in utterance
acoustic_params(:,26) = str2double(list_relative_pos);
% Relative position from end in utterance
acoustic_params(:,27) = str2double(list_relative_pos_end);

% Duration utt
[wav, fs] = audioread(sprintf('%sTEST%05d_syn.wav', path, ind_utt));
duration_utt = (length(wav)/fs) - 0.3;
acoustic_params(:,28) = duration_utt;
% Phon Rate
nbr_phon = len_utt - sum(ismember(list_char, list_pct));
acoustic_params(:,29) = ones(len_utt,1) * nbr_phon/duration_utt;
%  RMS Utt
acoustic_params(:,30) = ones(len_utt,1) * mean(all_rms(ismember(list_char, list_vowels)));
% acoustic_params(:,30) = 20*log10(str2double(list_rms_utt)/ref_level);
%  Mean Slope of vowels
acoustic_params(:,31) = ones(len_utt,1) * mean(all_slope(ismember(list_char, list_vowels)));
% Harmonic to noise ratio
% acoustic_params(:,32) = str2double(list_hnr);
acoustic_params(:,32) = ones(len_utt,1) * mean(str2double(list_hnr(ismember(list_char, list_vowels))));
% Mean F0 by vowels
acoustic_params(:,33) = ones(len_utt,1) * nanmean(linear_freq_to_semitone(str2double(list_f0_inter2(ismember(list_char, list_vowels))), ref_semiton));

% Local Pfitzinger
[local_pfitzinger, ~, ~] = compute_local_pfitzinger_by_utt(list_char, str2double(list_end) - str2double(list_start));
acoustic_params(:,34) = local_pfitzinger;

% Spectral balance (0-1000 / 1000-8000)
acoustic_params(:,35) = str2double(list_spectral_balance);

% % Pitch CWT
% if ~exist(sprintf('%sTEST%05d_syn_pitch_cwt.mat',path,ind_utt))
%     pitch_cwt_coef = nan(len_utt, nbr_pitch_cwt_coef);
%     return
% else
%     load(sprintf('%sTEST%05d_syn_pitch_cwt.mat',path,ind_utt)) % variable name pitch_cwt_mat
%     pitch_cwt_coef = pitch_cwt_mat';
% end
% acoustic_params(:,35:34+nbr_pitch_cwt_coef)= pitch_cwt_coef;

% % Delta F0
% acoustic_params(:,34) = ones(len_utt,1) * nanmean(linear_freq_to_semitone(str2double(list_f0_inter2(ismember(list_char, list_vowels))), ref_semiton));
% % Delta/Deltat F0
% acoustic_params(:,35) = ones(len_utt,1) * nanmean(linear_freq_to_semitone(str2double(list_f0_inter2(ismember(list_char, list_vowels))), ref_semiton));
end