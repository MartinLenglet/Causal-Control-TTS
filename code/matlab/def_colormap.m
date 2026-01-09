function custom_colormap = def_colormap(min_value, max_value)
    nbr_points = 100;
    custom_colormap = [];
    step = (max_value-min_value)/nbr_points;
    
%     max_value_color = 'blue';
    max_value_color = 'red';
    
    scale = min_value:step:max_value;
    for i_scale = scale
        if strcmp(max_value_color, 'blue')
            if i_scale<0
                custom_colormap = [custom_colormap; 1 max(1+i_scale, 0) max(1+i_scale, 0)];
            else
                custom_colormap = [custom_colormap; max(1-i_scale, 0) max(1-i_scale, 0) 1];
            end
        elseif strcmp(max_value_color, 'red')
            if i_scale<0
%                 custom_colormap = [custom_colormap; max(1+i_scale, 0) max(1+i_scale, 0) 1];
                custom_colormap = [custom_colormap; abs((i_scale-min_value)/min_value) abs((i_scale-min_value)/min_value) 1];
            else
%                 custom_colormap = [custom_colormap; 1 max(1-i_scale, 0) max(1-i_scale, 0)];
                custom_colormap = [custom_colormap; 1 abs((i_scale-max_value)/max_value) abs((i_scale-max_value)/max_value)];
            end
        end
    end
end