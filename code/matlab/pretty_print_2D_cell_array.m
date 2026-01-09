function cell_array = pretty_print_2D_cell_array(cell_array)
nbr_rows = size(cell_array, 1);
nbr_columns = size(cell_array, 2);

for row = 1:nbr_rows
   nbr_char_current_row = 0;
   for column = 1 :nbr_columns
       max_width_current_column = max(cellfun(@length, cell_array(:,column)));
       % Pad string
       current_cell = cell_array{row, column};
       if isempty(current_cell)
           current_cell = '';
       end
       fprintf('%s', pad(current_cell, max_width_current_column));
       nbr_char_current_row  = nbr_char_current_row + max_width_current_column;
       if column ~= nbr_columns
           fprintf(' | ');
           nbr_char_current_row  = nbr_char_current_row + 3;
       end
   end
   fprintf('\n');
   fprintf([repmat('.', 1, nbr_char_current_row), '\n']);
   fprintf('\n');
end
end