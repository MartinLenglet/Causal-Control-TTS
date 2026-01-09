function [gain_by_param_by_layer, r_square_by_param_by_layer, saturation_by_param_by_layer, figure_counter, max_by_param_by_layer] = compute_controllability_by_layer( ...
    list_models, ...
    params_to_control, ...
    std_by_param, ...
    name_layers, ...
    bias, ...
    ref_index, ...
    disp_all_graphs, ...
    figure_counter ...
    )

% Hyperparameter for graphs
frame_limits = [-3.5, 3.5];

nbr_models = length(list_models);
nbr_params = length(params_to_control(:,1));
nbr_layers = length(name_layers);

% Init results
gain_by_param_by_layer = zeros(nbr_params, nbr_layers, nbr_models);
r_square_by_param_by_layer = zeros(nbr_params, nbr_layers, nbr_models);
saturation_by_param_by_layer= zeros(nbr_params, nbr_layers, nbr_models);

max_by_param_by_layer = zeros(nbr_params, nbr_layers, nbr_models);

for i_param = 1:nbr_params
    current_param = params_to_control{i_param, 1};
    index_param = cell2mat(params_to_control(i_param, 2));
    current_std = std_by_param(index_param);

    if disp_all_graphs
        plotSettings;
        
        % Disp 1 graph by param
        figure(figure_counter);
        clf;
        fp = get(gcf,'Position');
        fp(3:4) = [1600 900];
        set(gcf,'Position',fp)
        set(gcf,'Renderer', 'painters');
        set(gcf,'PaperSize',[53 21])

        p = panel();
        p.pack(nbr_models, nbr_layers);
        m = 1;
        n = 1;
    end

    for i_model = 1:nbr_models
        % Load bias effect by model
        name_model = list_models{i_model};
        load(sprintf('data/controllability_array_%s', name_model)); % controllability_array

        for i_layer = 1:nbr_layers
            current_data = controllability_array(:, :, i_param, i_layer);
            % Normalize results (bricolé pour corriger la diff WAVEGLOW
            % HIFIGAN)
%             ref_results = current_data(:, ref_index);
            ref_results = nanmean(current_data(:, [ref_index-1, ref_index+1])')';
            nbr_bias = size(controllability_array, 2);
            for i_bias = 1:nbr_bias
                if i_bias == ref_index
                    current_data(:, i_bias) = current_data(:, i_bias) - current_data(:, i_bias);
                else
                    current_data(:, i_bias) = (current_data(:, i_bias) - ref_results)/current_std;
                end
            end
            nbr_points = size(current_data, 1)*size(current_data, 2);

            
            % Sigmoid fit

%             % non-centered on 0
%             ft = fittype('L + (U-L)/(1 + exp(-4*log(3)*(x-xmid)/xscale80))','indep','x'); 
% 
            % centered + symetric saturations
            ft = fittype('-U + 2*U/(1 + exp(-4*log(3)*(x)/xscale80))','indep','x');
            
%             % centered + lower and upper saturations can differ
%             ft = fittype('L + (U-L)/(1 + exp(-4*log(3)*(x)/xscale80))','indep','x'); 

            % filter nan
            x_table = reshape(kron(bias', ones(500, 1)), nbr_points, 1);
            y_table = reshape(current_data, nbr_points, 1);
            nanIndexes = isnan(y_table);
            x_table(nanIndexes) = [];
            y_table(nanIndexes) = [];

%             [mdl, gof] = fit(x_table, y_table, ft, 'start', [-2, 2, 4]);
            [mdl, gof] = fit(x_table, y_table, ft, 'start', [2, 4]);

            slope = log(3)* 2 * mdl.U / mdl.xscale80;
%             slope = log(3)* (mdl.U - mdl.L) / mdl.xscale80;
            max_range = max([mdl(-3), mdl(3)]);

            gain_by_param_by_layer(i_param, i_layer, i_model) = slope;
            r_square_by_param_by_layer(i_param, i_layer, i_model) = gof.rsquare;
            saturation_by_param_by_layer(i_param, i_layer, i_model) = mdl.U;
            max_by_param_by_layer(i_param, i_layer, i_model) = max_range;

            
            if disp_all_graphs
%                 subplot(nbr_models, nbr_layers, subplot_counter);

                p(m, n).select();
  %%           
%   cla
                % Plot range
                line(frame_limits,[1 1]*max([mdl(-3), mdl(3)]),'LineWidth',1,'LineStyle','--','Color',pcol.gipsaPSD); hold on
                line(frame_limits,-[1 1]*max([mdl(-3), mdl(3)]),'LineWidth',1,'LineStyle','--','Color',pcol.gipsaPSD)
                patch([frame_limits fliplr(frame_limits)],[-1 -1 1 1]*max_range,pcol.gipsaPSD,'LineStyle','none','FaceAlpha',0.025)
                
                % Plot boxes
                b=boxplot(current_data, bias, 'Positions', bias, 'Symbol', '','Color','k','Notch','on');
                set(b(6,:),'LineWidth',2,'Color','k')   % Medians in black
                h = findobj(gca,'Tag','Box');
                for j=1:length(h)
                    patch(get(h(j),'XData'),get(h(j),'YData'),pcol.gipsaPPC,'FaceAlpha',0.75);  % Paint boxes
                end
    
                % Linear fit
                plot(frame_limits, frame_limits,'Color',pcol.gipsaGAIA,'LineWidth',1);

                % Plot means of boxplot
                plot(bias, nanmean(current_data), 'k*');
                
                % Plot fit
                hf = plot(mdl);
                set(hf,'Color',pcol.gipsaPSD,'LineWidth',2);
                
                legend off;
                xlabel('');
                ylabel('');
                grid on;
                hold on;
                xlim(frame_limits);
                ylim(frame_limits);
                

                [m, n] = update_index_panel(m, n, nbr_layers);
                
                set(gca,'FontSize',12)
                if i_model == nbr_models
                    xlabel('Target bias (# std)')
                end
                if i_layer == 1
                    ylabel('Achieved bias (# std)')
                end
                text(0,3.25,['R_{\sigma}^{2} = ' sprintf('%0.2f | Control range = +/- %0.2f std',gof.rsquare, max_range)],'horizontalalignment','center','FontSize',12,'interpreter','tex')
                %                 xlabel(sprintf('%s | %s | R²=%0.2f | slope=%0.2f | L=%0.2f | U=%0.2f', name_model, name_layers{i_layer}, gof.rsquare, slope, -mdl.U, mdl.U));
%                 title(sprintf('%s | %s | R^{2}=%0.2f | slope=%0.2f', name_model, name_layers{i_layer}, gof.rsquare, slope),'interpreter','tex');
                title(sprintf('%s | layer: %s',strrep(list_models{i_model,2},'$',''), name_layers{i_layer}),'interpreter','tex','FontSize',12);

                xtickangle(60)
                
            end

            if i_model == 2 && i_param == 2 && (i_layer == 1 || i_layer == 3)

                figure(1000)

                plotSettings;
                posPlot = plotPosition(1,2,[0.0 0.18 0.07 0.14],[0.03 0.01 0.01 0.01]);
        
                % Disp 1 graph by param
                fp = get(gcf,'Position');
                fp(3:4) = [1450 650];
                set(gcf,'Position',fp)
                set(gcf,'Renderer', 'painters');
                set(gcf,'PaperSize',[51 23])

                switch(i_layer)
                    case 1
                        numplot = 1;
                    case 3
                        numplot = 2;
                end
                subplotMargin(posPlot,numplot);

                % Linear fit
                plot(frame_limits, frame_limits,'Color',pcol.gipsaGAIA,'LineWidth',1); hold on

                % Plot range background
                patch([frame_limits fliplr(frame_limits)],[-1 -1 1 1]*max_range,pcol.gipsaPSD,'LineStyle','none','FaceAlpha',0.025,'HandleVisibility','off')

                % Plot boxes
                b=boxplot(current_data, bias, 'Positions', bias, 'Symbol', '','Color','k','Notch','on');
                set(b(6,:),'LineWidth',2,'Color','k')   % Medians in black
                h = findobj(gca,'Tag','Box');
                for j=1:length(h)
                    patch(get(h(j),'XData'),get(h(j),'YData'),pcol.gipsaPPC,'FaceAlpha',0.75,'HandleVisibility','off');  % Paint boxes
                end

                xlim(frame_limits);
                ylim(frame_limits);
                
                % Plot means of boxplot
                plot(bias, nanmean(current_data), 'k*','HandleVisibility','off');

                % Plot fit
                hf = plot(mdl);
                set(hf,'Color',pcol.gipsaPSD,'LineWidth',2);

                % Plot range limits
                line(frame_limits,[1 1]*max([mdl(-3), mdl(3)]),'LineWidth',1,'LineStyle','--','Color',pcol.gipsaPSD); hold on
                line(frame_limits,-[1 1]*max([mdl(-3), mdl(3)]),'LineWidth',1,'LineStyle','--','Color',pcol.gipsaPSD,'HandleVisibility','off')

                legend off;
                xlabel('');
                ylabel('');
                grid on;
                hold on;

                
                % axis square
                set(gca,'FontName','Times','FontSize',24)
                xlabel('Target bias (# std)','FontName','Helvetica Neue','FontSize',28)
                if numplot == 1
                    ylabel('Achieved bias (# std)','FontName','Helvetica Neue','FontSize',28)
                end
                text(0,3.1,['R_{\sigma}^{2} = ' sprintf('%0.2f',gof.rsquare) ' | Control range = +/-' sprintf('%0.2f std', max_range)],'horizontalalignment','center','interpreter','tex','FontAngle','italic','FontName','Times','FontSize',26)
                %                 xlabel(sprintf('%s | %s | R²=%0.2f | slope=%0.2f | L=%0.2f | U=%0.2f', name_model, name_layers{i_layer}, gof.rsquare, slope, -mdl.U, mdl.U));
%                 title(sprintf('%s | %s | R^{2}=%0.2f | slope=%0.2f', name_model, name_layers{i_layer}, gof.rsquare, slope),'interpreter','tex');
                % title(sprintf('%s | layer: %s',strrep(list_models{i_model,2},'$',''), name_layers{i_layer}),'interpreter','latex','FontSize',28);
                title(sprintf('%s (%s)',list_models{i_model,2}, name_layers{i_layer}),'interpreter','latex','FontSize',34);

                xtickangle(60)

                lgnd = legend({'Optimal control';'Sigmoid regression';'Control range'},'FontName','Helvetica Neue','FontSize',24);
                lgnd.Position(1) = 0.82;
                lgnd.Position(2) = 0.45;
                % title(lgnd,'f_o control')

            end

        end
    end

%     saveas(gcf,cfg.results_root,'pdf');
    
    fprintf('\tControllability %s\n', current_param)
    if disp_all_graphs
        sgtitle(sprintf('Controllability %s', current_param));
        figure_counter = figure_counter + 1;
    end
end