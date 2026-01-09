"""Batch acoustic-parameter extraction for synthesized wavs.

Portable wrapper around the original paper script `calculate_acoustic_params_syn.praat`.

What it does
------------
For each subfolder listed in a text file, this script looks for:
  - TEST%05d_syn.wav
  - TEST%05d_seg.csv   (tab-separated table)
and writes:
  - TEST%05d_acoustic_params.csv

The output CSV uses a '|' separator and the exact header expected by the MATLAB
loader `matlab/load_acoustic_params.m`.

How to run (examples)
---------------------
macOS/Linux:
  praat --run praat/calculate_acoustic_params_syn_cli.praat \
    /abs/path/to/Embeddings_Visualization \
    praat/lists/list_folder_praat_script.txt \
    1000 1

Windows (PowerShell):
  & 'C:\\Program Files\\Praat.exe' --run praat\calculate_acoustic_params_syn_cli.praat `
    'C:\\path\\to\\Embeddings_Visualization' `
    'praat\\lists\\list_folder_praat_script.txt' `
    1000 1

Arguments
---------
1) base_folder: directory containing the subfolders listed in list_file
2) list_file: raw text file, one subfolder per line (relative to base_folder)
3) nbr_utt: number of utterances to iterate (e.g., 1000)
4) is_male: 1 for male pitch settings, 0 for female
"""

form Batch extraction settings
    sentence base_folder /path/to/Embeddings_Visualization
    sentence list_file praat/lists/list_folder_praat_script.txt
    positive nbr_utt 1000
    boolean is_male 1
endform

name_folder_global$ = base_folder
if right$ (name_folder_global$, 1) <> "/"
    name_folder_global$ = name_folder_global$ + "/"
endif

list_sub_folders$ = list_file
strings_object = Read Strings from raw text file: list_sub_folders$
numberOfSubFolder = Get number of strings

men_min_pitch = 70
men_max_pitch = 200
women_min_pitch = 150
women_max_pitch = 400

if is_male
    min_pitch_to_search = men_min_pitch
    max_pitch_to_search = men_max_pitch
else
    min_pitch_to_search = women_min_pitch
    max_pitch_to_search = women_max_pitch
endif

for ifile to numberOfSubFolder
    selectObject: strings_object
    sub_folder_name$ = Get string: ifile
    name_folder$ = name_folder_global$ + sub_folder_name$ + "/"

    for i_utt from 1 to nbr_utt
        if i_utt <10
            zeros$ = "0000"
        elsif i_utt <100
            zeros$ = "000"
        elsif i_utt < 1000
            zeros$ = "00"
        elsif i_utt < 10000
            zeros$ = "0"
        elsif i_utt < 100000
            zeros$ = ""
        endif

        name_file$ = "TEST" + zeros$ + string$ (i_utt)

        name_file_wav$ = name_folder$ + name_file$ + "_syn.wav"
        name_file_seg$ = name_folder$ + name_file$ + "_seg.csv"
        name_file_output$ = name_folder$ + name_file$ + "_acoustic_params.csv"

        if fileReadable (name_file_seg$)

            deleteFile: name_file_output$
            appendFile: name_file_output$, "character|start|end|gt_duration|z_score_gt|z_score_align|mean_f0_utt|std_f0_utt|f0_inter1|f0_inter2|f0_inter3|f1_inter1|f1_inter2|f1_inter3|f2_onset|f2_inter1|f2_inter2|f2_inter3|f3_inter1|f3_inter2|f3_inter3|rms|centerGravity|stdGravity|absolutePos|absolutePosEnd|relativePos|relativePosEnd|mean_rms_utt|HNR|spectral_balance"

            seg_object = Read Table from tab-separated file: name_file_seg$
            selectObject: seg_object
            nbr_line = Get number of rows

            wav_object = Read from file: name_file_wav$
            selectObject: wav_object
            pitch_object = To Pitch: 0,  min_pitch_to_search, max_pitch_to_search

            selectObject: wav_object
            formant_object = To Formant (burg): 0, 5, 5500, 0.025, 50

            selectObject: wav_object
            harmonicity_object = To Harmonicity (cc): 0.01, 75, 0.1, 1

            selectObject: pitch_object
            mean_pitch_utt_long = Get mean: 0, 0, "Hertz"
            std_pitch_utt_long = Get standard deviation: 0, 0, "Hertz"
            mean_pitch_utt$ = fixed$ (mean_pitch_utt_long, 6)
            std_pitch_utt$ = fixed$ (std_pitch_utt_long, 6)

            selectObject: wav_object
            rms_utt_long = Get root-mean-square: 0, 0
            rms_utt$ = fixed$ (rms_utt_long, 6)

            for i_char from 1 to nbr_line
                selectObject: seg_object
                current_char$ = Get value: i_char, "character"
                start_time = Get value: i_char, "start"
                start_time = start_time/1000
                end_time = Get value: i_char, "end"
                end_time = end_time/1000
                inter_time_1 = start_time + (end_time - start_time)/3
                inter_time_2 = start_time + 2*(end_time - start_time)/3

                gt_duration = Get value: i_char, "GTduration"
                gt_duration = gt_duration/1000

                z_score_gt = Get value: i_char, "ZScoreGT"
                z_score_align = Get value: i_char, "ZScoreAlign"

                absolute_pos = i_char
                absolute_pos_end = nbr_line - i_char
                relative_pos_long = i_char/nbr_line
                relative_pos$ = fixed$ (relative_pos_long, 6)
                relative_pos_end_long = (nbr_line - i_char)/nbr_line
                relative_pos_end$ = fixed$ (relative_pos_end_long, 6)

                if start_time = 0 and end_time = 0 or abs(start_time-end_time)<0.01161
                    mean_pitch_inter1$ = "--undefined--"
                    mean_pitch_inter2$ = "--undefined--"
                    mean_pitch_inter3$ = "--undefined--"
                    formant_1_inter1$ = "--undefined--"
                    formant_1_inter2$ = "--undefined--"
                    formant_1_inter3$ = "--undefined--"
                    formant_2_onset$ = "--undefined--"
                    formant_2_inter1$ = "--undefined--"
                    formant_2_inter2$ = "--undefined--"
                    formant_2_inter3$ = "--undefined--"
                    formant_3_inter1$ = "--undefined--"
                    formant_3_inter2$ = "--undefined--"
                    formant_3_inter3$ = "--undefined--"
                    rms$ = "--undefined--"
                    center_gravity$ = "--undefined--"
                    std_gravity$ = "--undefined--"
                    hnr$ = "--undefined--"
                    spectral_balance$ = "--undefined--"
                else
                    selectObject: pitch_object
                    mean_pitch_inter1_long = Get mean: start_time, inter_time_1, "Hertz"
                    mean_pitch_inter1$ = fixed$ (mean_pitch_inter1_long, 6)

                    mean_pitch_inter2_long = Get mean: inter_time_1, inter_time_2, "Hertz"
                    mean_pitch_inter2$ = fixed$ (mean_pitch_inter2_long, 6)

                    mean_pitch_inter3_long = Get mean: inter_time_2, end_time, "Hertz"
                    mean_pitch_inter3$ = fixed$ (mean_pitch_inter3_long, 6)

                    selectObject: formant_object
                    formant_1_inter1_long = Get mean: 1, start_time, inter_time_1, "hertz"
                    formant_1_inter1$ = fixed$ (formant_1_inter1_long, 6)
                    formant_2_inter1_long = Get mean: 2,  start_time, inter_time_1, "hertz"
                    formant_2_inter1$ = fixed$ (formant_2_inter1_long, 6)
                    formant_3_inter1_long = Get mean: 3,  start_time, inter_time_1, "hertz"
                    formant_3_inter1$ = fixed$ (formant_3_inter1_long, 6)

                    formant_1_inter2_long = Get mean: 1, inter_time_1, inter_time_2, "hertz"
                    formant_1_inter2$ = fixed$ (formant_1_inter2_long, 6)
                    formant_2_inter2_long = Get mean: 2,  inter_time_1, inter_time_2, "hertz"
                    formant_2_inter2$ = fixed$ (formant_2_inter2_long, 6)
                    formant_3_inter2_long = Get mean: 3,  inter_time_1, inter_time_2, "hertz"
                    formant_3_inter2$ = fixed$ (formant_3_inter2_long, 6)

                    formant_1_inter3_long = Get mean: 1, inter_time_2, end_time, "hertz"
                    formant_1_inter3$ = fixed$ (formant_1_inter3_long, 6)
                    formant_2_inter3_long = Get mean: 2,  inter_time_2, end_time, "hertz"
                    formant_2_inter3$ = fixed$ (formant_2_inter3_long, 6)
                    formant_3_inter3_long = Get mean: 3,  inter_time_2, end_time, "hertz"
                    formant_3_inter3$ = fixed$ (formant_3_inter3_long, 6)

                    formant_2_onset_long = Get value at time: 2, start_time, "hertz", "Linear"
                    formant_2_onset$ = fixed$ (formant_2_onset_long, 6)

                    selectObject: wav_object
                    rms_long = Get root-mean-square: start_time, end_time
                    rms$ = fixed$ (rms_long, 6)

                    selectObject: wav_object
                    subWav_object = Extract part: start_time, end_time, "rectangular", 1, "no"
                    selectObject: subWav_object
                    subSpectrum_object = To Spectrum: "no"
                    selectObject: subSpectrum_object
                    center_gravity_long = Get centre of gravity: 2
                    center_gravity$ = fixed$ (center_gravity_long, 6)
                    std_gravity_long = Get standard deviation: 2
                    std_gravity$ = fixed$ (std_gravity_long, 6)

                    spectral_balance_long = Get band energy difference: 0, 1000, 1000, 8000
                    spectral_balance$ = fixed$ (spectral_balance_long, 6)

                    selectObject: harmonicity_object
                    hnr_long = Get mean: start_time, end_time
                    hnr$ = fixed$ (hnr_long, 6)

                    selectObject: subWav_object
                    Remove
                    selectObject: subSpectrum_object
                    Remove
                endif

                appendFile: name_file_output$, newline$, current_char$, "|", start_time, "|", end_time, "|", gt_duration, "|", z_score_gt, "|", z_score_align, "|", mean_pitch_utt$, "|", std_pitch_utt$, "|", mean_pitch_inter1$, "|", mean_pitch_inter2$, "|", mean_pitch_inter3$, "|", formant_1_inter1$, "|", formant_1_inter2$, "|", formant_1_inter3$, "|", formant_2_onset$, "|", formant_2_inter1$, "|", formant_2_inter2$, "|", formant_2_inter3$, "|", formant_3_inter1$,"|", formant_3_inter2$,"|", formant_3_inter3$,"|",rms$,"|", center_gravity$,"|",std_gravity$,"|", absolute_pos,"|", absolute_pos_end,"|", relative_pos$,"|", relative_pos_end$,"|",rms_utt$,"|",hnr$,"|",spectral_balance$
            endfor

            selectObject: seg_object
            Remove
            selectObject: wav_object
            Remove
            selectObject: pitch_object
            Remove
            selectObject: formant_object
            Remove
            selectObject: harmonicity_object
            Remove

        endif
    endfor
endfor

selectObject: strings_object
Remove
