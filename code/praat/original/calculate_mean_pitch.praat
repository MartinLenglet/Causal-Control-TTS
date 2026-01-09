name_folder$ = "/home/lengletm/Documents/Etudes/Encoder_Embeddings/Embeddings_Visualization/expe_punctuation_updated/_syn_baseline/"
# nbr_utt = 29557
nbr_utt = 137

name_file_output$ = name_folder$ + "SYN_9_start_3_f0.csv"
deleteFile: name_file_output$
appendFile: name_file_output$, "mean|std|slope"
part_audio = 3

for i_utt from 1 to nbr_utt
	name_file$ = "CIBLE" + string$ (i_utt) + "_9"
	name_file_wav$ = name_folder$ + name_file$ + "_syn.wav"

	wav_object = Read from file: name_file_wav$
	selectObject: wav_object
	
	total_duration = Get total duration
	# sub_wav_object = Extract part: 0, total_duration/part_audio, "rectangular", 1, "no"
	# sub_wav_object = Extract part: (part_audio-1)*total_duration/part_audio, total_duration, "rectangular", 1, "no"
	# selectObject: sub_wav_object

	pitch_object = To Pitch: 0,  75, 600
	selectObject: pitch_object

	# current_mean_long = Get mean: 0, 0, "semitones re 1 Hz"
	# current_std_long = Get standard deviation: 0, 0, "semitones"

	current_mean_long = Get mean: 0, total_duration/part_audio, "semitones re 1 Hz"
	current_std_long = Get standard deviation: 0, total_duration/part_audio, "semitones"

	#current_mean_long = Get mean: (part_audio-1)*total_duration/part_audio, total_duration, "semitones re 1 Hz"
	#current_std_long = Get standard deviation: (part_audio-1)*total_duration/part_audio, total_duration, "semitones"

	current_slope_long = Get mean absolute slope: "Semitones"

	current_mean$ = fixed$ (current_mean_long, 2)
	current_std$ = fixed$ (current_std_long, 2)
	current_slope$ = fixed$ (current_slope_long, 2)

	appendFile: name_file_output$, newline$, current_mean$,"|",current_std$,"|",current_slope$

	selectObject: wav_object
	Remove
	selectObject: pitch_object
	Remove
	# selectObject: sub_wav_object
	# Remove
endfor