%% Paramterstudy for CLNN
clear; clc; close all;
spath = 'Results\';

%% Fixed parameters
% Data
data.init = load('Data\init.mat');
data.cl = load('Data\A.mat');
params.num_out = 1;

% Support-Points for evaluation
eval_points = load('Data\eval_pointsA.mat');

% Initial Training
params.init_opts = trainingOptions( 'adam', ...
                                    'MaxEpochs',50,...
                                    'MiniBatchSize', 32, ...
                                    'InitialLearnRate',1e-3, ...
                                    'LearnRateSchedule', 'piecewise',...
                                    'LearnRateDropFactor', 0.7,...
                                    'LearnRateDropPeriod', 5,...
                                    'GradientDecayFactor', 0.9,...
                                    'SquaredGradientDecayFactor', 0.9,...
                                    'Verbose',true, ...
                                    'Plots','training-progress');
params.eval_split = 0.2;

% Specific parameters
params.net_size = [132, 92];
params.bins_per_dim = [0.2, 0.2];
params.bins_depth = 5;

%% Varied parameters
params.l = {10; 20; 30};
params.ds = {150; 200; 250};
params.hf = {0; 0.05; 0.10; 0.20; 0.50};
params.delta_theta = {0.01; 0.05; 0.10; []};

configs = fullfact([numel(params.l), ...
                    numel(params.ds), ...
                    numel(params.hf), ...
                    numel(params.delta_theta)]);
configs = sortrows(configs);

%% Iterate on configurations
l_old = NaN;
for ii = 1:size(configs,1)
    % Get current parameters
    l = params.l{configs(ii,1)};
    ds = params.ds{configs(ii,2)};
    hf = params.hf{configs(ii,3)};
    delta_theta = params.delta_theta{configs(ii,4)};
    % Name config
    name = replace(sprintf('l_%03d_ds_%03d_hf_%.2f_dt_%.2f', l, ds, hf, delta_theta), '.','p');
    
    % ------------------------------------------------------------------- %
    % I) Initial Training
    % ------------------------------------------------------------------- %
    % Perform initial training if necessary only
    if ~isequal(l, l_old)
        % Initial Training
        myCLNN = CLNN([ data.init.data.info.x_limits; ...
                        data.init.data.info.y_limits], ... 
                        params.num_out,            ... 
                        [params.net_size, l],           ... 
                        params.bins_per_dim,     ...
                        params.bins_depth          ...
                        );
        [RMSE_init_train, RMSE_init_eval]  = ...
            myCLNN.init_train([ data.init.data.out.x, ...
                                data.init.data.out.y], ...
                                data.init.data.out.z , ...
                                params.eval_split, ...
                                params.init_opts);
                            
        % Check reults
        z_init_pred = myCLNN.init_pred([ data.init.data.out.x, ...
                                         data.init.data.out.y]);
        figure;
        hold on
        scatter3(data.init.data.out.x, data.init.data.out.y, data.init.data.out.z, 'xb', 'DisplayName', 'Ground truth');
        scatter3(data.init.data.out.x, data.init.data.out.y, z_init_pred, 'or', 'DisplayName', 'Prediction');
        xlabel('x'); ylabel('y'), zlabel('z');
        hold off
        grid on;
        axis equal;
        legend()
        if ~exist(spath, 'dir')
            mkdir(spath)
        end
        saveas(gcf, strcat(spath, 'init', name), 'fig');
        close
    end
    
    % Reset to state after initial training
    myCLNN.reset()
                
    % ------------------------------------------------------------------- %
    % II) Continual Training
    % ------------------------------------------------------------------- %
    t = size(data.cl.data.out.x,1);

    % Preallocation 
    gR_cl = zeros(t,1);
    gR_init = zeros(t,1);
    eOP_cl = zeros(t,1);
    
    % Evaluation on eval grid
    z_pred_eval_points_init = myCLNN.init_pred(eval_points.xy);
    z_pred_eval_points_cl = z_pred_eval_points_init; % since no update yet
    eOP_init = abs(data.cl.data.out.z-myCLNN.init_pred([data.cl.data.out.x, data.cl.data.out.y]));
    
    % Iterate on time steps
    for jj = 1:t
        % Do not use CLNN.cl_eval or CLNN.init_eval at each time step to
        % calculate the RMSE because these methods call the prediction methods,
        % which are slow. Since the predictions on the eval-points do not change
        % every time step, calculation of the RMSE is performed manually to save 
        % computation time by reducing the call of the prediction methods. 

        % Evaluate CLNN
        gR_init(jj) = sqrt(mean((z_pred_eval_points_init - eval_points.z(:,jj)).^2));
        gR_cl(jj) = sqrt(mean((z_pred_eval_points_cl - eval_points.z(:,jj)).^2));
        
        % Save state in time step 50,000
        if jj == 50000
            state50000.eval_points.xy = eval_points.xy;
            state50000.target = eval_points.z(:,jj);
            state50000.pred_init = z_pred_eval_points_init;
            state50000.pred_cl = z_pred_eval_points_cl;
        end

        % Perform update every ds time step
        if and( mod(jj, ds) == 0, jj+ds<=t) 
           
            % Current data
            x_new = data.cl.data.out.x(jj-ds+1:jj,:);
            y_new = data.cl.data.out.y(jj-ds+1:jj,:);
            z_new = data.cl.data.out.z(jj-ds+1:jj,:);
            
            % Prediction on Operating Point
            eOP_cl(jj-ds+1:jj) = abs(z_new-myCLNN.cl_pred([x_new, y_new]));
            
            % Update CLNN
            myCLNN.cl_update(   [x_new, y_new], ...
                                z_new,          ...
                                hf,             ... 
                                delta_theta);
                            
            % Update predictions on eval grid                            
            z_pred_eval_points_cl = myCLNN.cl_pred([eval_points.xy]);
        end

    end
    
    % Store results
    save(   strcat(spath, name, '.mat'), ...
            'RMSE_init_train', 'RMSE_init_eval', ...
            'gR_cl', 'gR_init', 'eOP_cl', 'eOP_init', ...
            'state50000');
    
    l_old = l;
end
