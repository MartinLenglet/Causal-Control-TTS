function write_residual_OP(filename, predicted_acoustic_params,considered_acoustic_params,calc_acoust,label_acoustic,label_layer,label_model,label_encdec,corr_coef)
    
% Get columns and sort (sorting is optional)
    [~,indsort] = sort(considered_acoustic_params);
    x = (considered_acoustic_params(indsort));
    y = (predicted_acoustic_params(indsort));
    res = y-x;
    
    % Format data into vectors
    Nobs = length(y);
    obs_col = (1:Nobs)';
    layer_col = repmat(label_layer,Nobs,1);
    acoust_col = repmat(label_acoustic,Nobs,1);
    model_col = repmat(label_model,Nobs,1);
    encdec_col = repmat(label_encdec,Nobs,1);
    corr_col = repmat(corr_coef,Nobs,1);
    calc_col = repmat(calc_acoust,Nobs,1);

    % Convert data to a table
    T = table(obs_col, acoust_col, calc_col, model_col, encdec_col, layer_col, corr_col, res);
    writetable(T, filename, 'WriteMode', 'append');
    
    % Plot
    if 0
        %%
        figure(1000); clf
        subplot(221)
        plot(x,'.'); hold on
        plot(y,'.')
        grid on
        title(sprintf('Prediction of %s from layer %s (R^2 = %0.2f)',label_acoustic,label_layer,corr_coef))
    
        subplot(223)
        plot((y-x),'.')
        grid on
    
        subplot(224)
        boxplot(y-x,'position',1); hold on
        boxplot(sqrt((y-x).^2),'position',2)
        grid on
        xlim([0 3])
        ylim([min(y-x) max(sqrt((y-x).^2))])
    end
end