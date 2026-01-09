function [corr_coef, predicted_acoustic_params, all_beta, all_beta_normalized, all_bias, all_theta, correlation_by_dim, all_beta_acoustic] = acoustic_regression_in_latent_space(coord, acoustic_params, selected_dim, reduce_model)
    % Trim nan acoustic data
    nan_index = isnan(acoustic_params);
    trim_acoustic_params = acoustic_params;
    trim_coord = coord;
    trim_acoustic_params(sum(nan_index, 2) > 0, :) = [];
    trim_coord(sum(nan_index, 2) > 0, :) = [];
    nbr_points_trim = length(trim_coord(:,1));
    
    % Number of dimensions to keep for regression
    min_dim_by_regression = 10;
    max_correlation_reduction_percentage = 0.01;
    min_residual_reduction_percentage = 0.01;

    % Use to reduce model to limited number of dim (better results but
    % takes a long time)
%     reduce_model = false;

    nbr_acoustic_params = length(acoustic_params(1,:));
    nbr_points = length(coord(:,1));
    nbr_dim = length(coord(1,:));
    
    corr_coef = zeros(1, nbr_acoustic_params);
    err_by_param = zeros(1, nbr_acoustic_params);
    predicted_acoustic_params = zeros(nbr_points, nbr_acoustic_params);
    all_bias = zeros(1, nbr_acoustic_params);
    all_theta = zeros(1, nbr_acoustic_params);
    
    all_beta = zeros(nbr_dim, nbr_acoustic_params);
    all_beta_acoustic = zeros(nbr_acoustic_params, nbr_acoustic_params);
    all_beta_normalized = zeros(nbr_dim, nbr_acoustic_params);
    correlation_by_dim = zeros(nbr_dim, nbr_acoustic_params);
    
%     reduced_acoustic_params = acoustic_params;
%     mean_acoustic_params = nanmean(acoustic_params);
%     std_acoustic_params = nanstd(acoustic_params);
%     for i_points = 1:nbr_points
%         reduced_acoustic_params(i_points, :) = (reduced_acoustic_params(i_points, :) - mean_acoustic_params)./std_acoustic_params;
%     end
        
    for i_acoustic = 1:nbr_acoustic_params
%         copy_coord = coord;
        
        % Use other acoustic params as data for regression
%         other_acoustic_params = reduced_acoustic_params;
        
%         other_acoustic_params(:, i_acoustic) = zeros(nbr_points, 1);
%         other_acoustic_params =  other_acoustic_params(:, 1);
%         other_acoustic_params(:, i_acoustic) = [];
        
%         copy_coord = [copy_coord, other_acoustic_params];
%         [Mdl,FitInfo] = fitrlinear(copy_coord, acoustic_params(:,i_acoustic), 'Learner', 'leastsquares');
%         [Mdl,FitInfo] = fitrlinear(copy_coord, acoustic_params(:,i_acoustic), 'Learner', 'leastsquares', 'Solver', 'lbfgs');
        

        % Fit hyperplans between latent space a + nbr_acoustic_params - 1nd current acoustic
        % parameter (prediction)
%         [Mdl_best,FitInfo_best] = fitrlinear(coord, acoustic_params(:,i_acoustic), 'Learner', 'leastsquares', 'Solver', 'lbfgs');
        [Mdl_best,FitInfo_best] = fitrlinear(trim_coord, trim_acoustic_params(:,i_acoustic), 'Learner', 'leastsquares', 'Solver', 'lbfgs');
        
        % Prediction on syn parameters
        YHat_best = predict(Mdl_best,coord);
        YHat_trimmed_best = predict(Mdl_best,trim_coord);
        
%         [R, ~] = corrcoef([YHat_best, acoustic_params(:,i_acoustic)],'Rows','complete');
%         best_corr_coef = R(1,2);
        best_corr_coef = 1 - sum((YHat_trimmed_best-trim_acoustic_params(:,i_acoustic)).^2)/(var(trim_acoustic_params(:,i_acoustic))*length(trim_acoustic_params(:,i_acoustic)));
        best_err_by_param = immse(YHat_best, acoustic_params(:,i_acoustic));
        
        % Search subset of dimensions for each acoustic param
        if reduce_model
            mask = [];
            while nbr_dim - length(mask) > min_dim_by_regression
                corr_coef_test_by_dim = zeros(1, nbr_dim);
                error_by_param_test_by_dim = zeros(1, nbr_dim);
                for i_dim = 1:nbr_dim
                    if ~ismember(i_dim, mask)
                        % Test current masking
                        mask_test = [mask, i_dim];
                        coord_test = coord;
                        coord_test(:, mask_test) = zeros(nbr_points, length(mask_test));
                        trim_coord_test = trim_coord;
                        trim_coord_test(:, mask_test) = zeros(nbr_points_trim, length(mask_test));

                        % Fit Model
                        [Mdl_test,FitInfo_test] = fitrlinear(trim_coord_test, trim_acoustic_params(:,i_acoustic), 'Learner', 'leastsquares', 'Solver', 'lbfgs');
                        YHat_test = predict(Mdl_test,coord_test);
                        YHat_trimmed_test = predict(Mdl_test,trim_coord_test);

                        % Test Model
%                         [R, ~] = corrcoef([YHat_test, acoustic_params(:,i_acoustic)],'Rows','complete');
%                         corr_coef_test_by_dim(i_dim) = R(1,2);
                        corr_coef_test_by_dim(i_dim) = 1 - sum((YHat_trimmed_test-trim_acoustic_params(:,i_acoustic)).^2)/(var(trim_acoustic_params(:,i_acoustic)) * length(trim_acoustic_params(:,i_acoustic)));
                        error_by_param_test_by_dim(i_dim) = immse(YHat_test, acoustic_params(:,i_acoustic));
                    else
                        corr_coef_test_by_dim(i_dim) = -inf;
                        error_by_param_test_by_dim(i_dim) = inf;
                    end
                end
                % Suppr least impactfull dimension
                [max_reduced_correlation, index_least_impactfull_correlation] = max(corr_coef_test_by_dim);
                [min_reduced_residual, index_least_impactfull_residual] = min(error_by_param_test_by_dim);

                if abs(max_reduced_correlation-best_corr_coef)/best_corr_coef < max_correlation_reduction_percentage
                    mask = [mask, index_least_impactfull_correlation];

                    fprintf('Acoustic regression: %d/%d | Dim: %d/%d | Max Corr: %.04f/%.04f\n', i_acoustic,  nbr_acoustic_params, nbr_dim-length(mask), nbr_dim, max_reduced_correlation, best_corr_coef);
                else
                    break;
                end

    %             if  abs(min_reduced_residual-best_err_by_param)/best_err_by_param < min_residual_reduction_percentage
    %                 mask = [mask, index_least_impactfull_residual];
    %                 
    %                 fprintf('Acoustic regression: %d/%d | Dim: %d/%d | Min Residual: %.04f/%.04f\n', i_acoustic,  nbr_acoustic_params, nbr_dim-length(mask), nbr_dim, min_reduced_residual, best_err_by_param);
    %             else
    %                 break;
    %             end
            end

            % Apply Mask
            masked_coord = coord;
            masked_coord(:, mask) = zeros(nbr_points, length(mask));
            trim_masked_coord = trim_coord;
            trim_masked_coord(:, mask) = zeros(nbr_points_trim, length(mask));

            [Mdl,FitInfo] = fitrlinear(trim_masked_coord, trim_acoustic_params(:,i_acoustic), 'Learner', 'leastsquares', 'Solver', 'lbfgs');
            YHat = predict(Mdl,masked_coord);
            YHat_trimmed = predict(Mdl,trim_masked_coord);

%             [R, ~] = corrcoef([YHat, acoustic_params(:,i_acoustic)],'Rows','complete');
%             corr_coef(i_acoustic) = R(1,2);
            corr_coef(i_acoustic) = 1 - sum((YHat_trimmed-trim_acoustic_params(:,i_acoustic)).^2)/(var(trim_acoustic_params(:,i_acoustic))*length(trim_acoustic_params(:,i_acoustic)));
            err_by_param(i_acoustic) = immse(YHat, acoustic_params(:,i_acoustic));
        else
            Mdl = Mdl_best;
            FitInfo = FitInfo_best;
            YHat = YHat_best;
            
            corr_coef(i_acoustic) = best_corr_coef;
            err_by_param(i_acoustic) = best_err_by_param;
        end
        
        predicted_acoustic_params(:, i_acoustic) = YHat;
        current_beta = Mdl.Beta;
        all_beta(:,i_acoustic) = current_beta;
%         all_beta_acoustic(:,i_acoustic) = current_beta(nbr_dim+1:end);
        all_beta_normalized(:,i_acoustic) = current_beta/(norm(current_beta)^2);
        current_bias = Mdl.Bias;
        all_bias(i_acoustic) = current_bias;
        
%         for i_dim = 1:nbr_dim
%             [Mdl,FitInfo] = fitrlinear(copy_coord(:,i_dim), acoustic_params(:,i_acoustic), 'Learner', 'leastsquares', 'Solver', 'lbfgs');
%             YHat = predict(Mdl,copy_coord(:,i_dim));
%             [R, ~] = corrcoef([YHat, acoustic_params(:,i_acoustic)],'Rows','complete');
%             correlation_by_dim(i_dim,i_acoustic) = R(1,2);
%         end
        
        % Get theta for current dim
        if selected_dim
            theta = atan(current_beta(selected_dim(2))/current_beta(selected_dim(1)));
            if (current_beta(1) < 0)
                theta = theta + pi;
            end
            all_theta(i_acoustic) = theta;
        end
    end
end