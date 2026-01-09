function char_cell_array = phon_input_to_cell_array(phon_input)
    nbr_input = length(phon_input);
    char_cell_array = [];
    is_phon = false;
    current_cell = '';
    for i_input = 1:nbr_input
        current_phon = phon_input(i_input);
        if is_phon
            if current_phon == '}'
                is_phon = false;
                char_cell_array = [char_cell_array; {current_cell}];
                current_cell = '';
            elseif current_phon == ' '
                char_cell_array = [char_cell_array; {current_cell}];
                current_cell = '';
            else
                current_cell = [current_cell current_phon];
            end
        else
            if current_phon == '{'
                is_phon = true;
            else
                char_cell_array = [char_cell_array; {current_phon}];
            end
        end
    end
end