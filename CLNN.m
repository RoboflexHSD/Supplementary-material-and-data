%% Continual Learning Neural Network (CLNN)    
% CLNN class to apply continual model update for artificial neural 
% regression networks.
% Use this class to: 
%
% * implement an artificial neural network (ANN) model object for 
% multidimensional regression
% * perform an initial training for this ANN with a representative
%   dataset for the whole feature space using backpropagation
% * perform model updates on a non-representative subset of new data
% * make predictions on initial or current model
%
% *Note*: _The neural networks should contain at least 2 hidden layers,
% because the last hidden layer does not contain an activation function and
% therefore only a linear mapping is possible when using only one hidden
% layer. The reason for omitting the activation function in the last hidden
% layer is that the ReLU function is used as activation function and
% therefore only positive outputs would be possible during intial training,
% since the weights in the output layer are frozen to one (positive weights
% only!)._
%
% *This class requires the MATLAB Deep Learning Toolbox and MATLAB 
% Optimization Toolbox!*
%
%%

classdef CLNN < handle    
    %% Properties 
    % * _Architecture_: Struct containing information on the ANNs
    % architecture; *SetAccess*: immutable; *GetAccess*: public
    % * _Buffer_: Gridded Ring Buffer (GRB) object to manage historical data;
    % *SetAccess*: protected, *GetAccess*: public
    % * _BufferInitialized_: Copy of _Buffer_ that is created after 
    % initialization. Needed for resetting to the state after initial 
    % training; *SetAccess*: private, *GetAccess*: private  
    % * _Info_: Struct containing information about changes of the object; 
    % *SetAccess*: protected, *GetAccess*: public
    % * _Initialization_: Struct containing information about the initial
    % training of the Neural Network; *SetAccess*: protected, *GetAccess*: 
    % public
    % * _latest_updated_bins_: Array containing indices (columns) of bins
    % (rows) that should be excluded when calling the get-method of the GRB;
    % *SetAccess*: private, *GetAccess*: private
    % * _layers_: Layer defining the ANNs structure; *SetAccess*:
    % immutable, *GetAccess*: private 
    % * _Normalization_: Struct containing information about the
    % normalization of the input and output data; *SetAccess*: private,
    % *GetAccess*: private 
    % * _thetas_: Vector containing the weights, which are updated during
    % the continual learning phase; *SetAccess*: private, *GetAccess*:
    % private 
    % * _Updates_: Table containing information about the model updates 
    % during the continual learning phase; *SetAccess*: protected,
    % *GetAccess*: public 
    
    properties(SetAccess = immutable, GetAccess = public)
        Architecture = struct(  'num_inputs', NaN, ...
                                'num_outputs', NaN, ...
                                'hidden_layer', NaN...
                                );
    end
    
    properties(SetAccess = protected, GetAccess = public)
        Buffer = GRB.empty;
        Initialization = struct('net', NaN, ...
                                'num_train', NaN, ...
                                'num_eval', NaN, ...
                                'trainingoptions', NaN, ...
                                'RMSE_train', NaN, ...
                                'RMSE_eval', NaN, ...
                                'RMSE_train_N', NaN, ...
                                'RMSE_eval_N', NaN ...
                                );      
        Updates = table('Size',[0, 4], ...
                        'VariableTypes', {'double', 'double', 'double', 'double'},...
                        'VariableNames',{'No', 'ds', 'hf', 'deltatheta'} ...
                        );
        Info = struct(  'Build', NaN, ...
                        'Initialization', NaN, ...
                        'LatestUpdate', NaN, ...
                        'LatestReset', NaN ...
                        );
    end
   
    properties(SetAccess = immutable, GetAccess = private)
        layers = nnet.cnn.layer.Layer.empty;
    end
    
    properties(SetAccess = private, GetAccess = private)
        Normalization = struct( 'C_in', NaN, ...
                                'S_in', NaN, ...
                                'C_out', NaN, ...
                                'S_out', NaN ...
                                );         
        thetas;
        latest_updated_bins;
        BufferInitialized = GRB.empty;
    end
    
    methods
        %% Constructor
        function obj = CLNN(x_limits, num_outputs, hidden_layer, approx_grid_dx, samples_per_bin)
            %%%
            %   obj = CLNN(x_limits, num_outputs, hidden_layer, approx_grid_dx, samples_per_bin) 
            %%%
            %
            % Creates an object of CLNN with non-trained ANN model and
            % empty GRB. 
            %
            % Arguments:
            %
            % * _x_limits_: 2D-array containing the lower (first column)
            % and upper (second column) boundaries of the considered input
            % space: [x1_min, x1_max; ...].
            % * _num_outputs_: Number of model outputs.
            % * _hidden_layer_ [optional]: Nodes per layers. 
            % Default: [132, 92, 30]
            % * _approx_grid_dx_ [optional]: Approx. grid spacing for each
            % dimension of the input space:discretization in x1, x2, ...
            % Default: diff(x_limits, [], 2)/10
            % * _samples_per_bin_ [optional]: number of samples per bin. 
            % Default: 5
            %
            % Returns: 
            %
            % * _CLNN-object_
            %
            %%%
            
            arguments
                x_limits (:,2) double                                                   
                num_outputs (1,1) {mustBePositive, mustBeInteger}
                hidden_layer (1,:) {mustBePositive, mustBeInteger} = [132, 92, 30];
                approx_grid_dx (:,1) double {mustBePositive} = diff(x_limits, [], 2)/10;
                samples_per_bin (1,1) {mustBePositive, mustBeInteger} = 5;
            end
            assert(all(x_limits(:,1)<x_limits(:,2), 'all'), ...
                'Lower limits in ''x_limits'' (first column) must have lower values than upper limits (second column)!')
            if sum(approx_grid_dx>diff(x_limits, [], 2)/2)>0
                warning("Check discretization: At least one dimension has one bin only.")
            end
          
            % Set properties
            obj.Architecture.num_inputs = size(x_limits,1);
            obj.Architecture.num_outputs = num_outputs;
            obj.Architecture.hidden_layer = hidden_layer;            
            obj.Info.Build = datetime;
            
            % Setup net (stack layer)
            obj.layers(1) = featureInputLayer(  obj.Architecture.num_inputs, ...            % Input-layer
                                                'Name', 'input', ...                        %       .
                                                'Normalization', 'none' ...                 %       .
                                                );                                          %       .
            c = 1;                                                                          % Loop to stack inner layers with activations:
            for ii = 2:2:numel(obj.Architecture.hidden_layer)*2-1                           % (-1) in for excludes the last FC-Layer!
                obj.layers(ii) = fullyConnectedLayer(obj.Architecture.hidden_layer(c), ...  %   - FC-Layer c
                                                        'Name', sprintf('fc%i',c) ...       %       .
                                                        );                                  %       .
                obj.layers(ii+1) = reluLayer('Name', sprintf('activation%i',c));            %   - Activation c                  
                c = c+1;                                                                    %       .
            end                                                                             %       .
            if isempty(ii)
                ii = 0;
            end
            obj.layers(ii+2) = fullyConnectedLayer( obj.Architecture.hidden_layer(end), ... % Next to last FC-Layer (without activation!)
                                                    'Name', sprintf('fc%i',c));             %       .
            obj.layers(ii+3) = fullyConnectedLayer( obj.Architecture.num_outputs, ...       % Last FC-Layer with thetas / Output-layer
                                                    'Name', 'thetas', ...                   %       .   
                                                    'WeightsInitializer', 'ones', ...       %       .
                                                    'BiasInitializer', 'zeros', ...         %       .
                                                    'WeightLearnRateFactor', 0, ...         %       .
                                                    'BiasLearnRateFactor', 0 ...            %       .
                                                    );                                      %       .
            obj.layers(ii+4) = regressionLayer('Name', 'output');                           % Regression output-layer (needed in MATLAB)
            
            % Setup GRB
            obj.Buffer = GRB(x_limits, num_outputs, approx_grid_dx, samples_per_bin);      
      end
      
      %% Methods
      function [RMSE_train, RMSE_eval, ...
                RMSE_train_N, RMSE_eval_N] = init_train(obj, x, y, eval_split, options)
            %%% 
            %   [RMSE_train, RMSE_eval, RMSE_train_N, RMSE_eval_N] =
            %   init_train(obj, x, y, eval_split, options) 
            %%%
            %
            % Performs initial training on static dataset using
            % backpropagation. 
            %
            % Arguments: 
            %
            % * _x_: Inputs (rows: samples; cols: features).
            % * _y_: Outputs (rows: samples; cols: outputs).
            % * _eval_split_ [optional]: fraction of data separated for
            % validation. 
            % Default: 0
            % * _options_ [optional]: trainingOptions-object 
            %   (nnet.cnn.TrainingOptionsADAM or 
            %    nnet.cnn.TrainingOptionsRMSProp or 
            %    nnet.cnn.TrainingOptionsSGDM). 
            % Note: For detailed setting of training parameters 
            % [trainingOptions()], see MATLAB Deep Learning Toolbox 
            % Documentation.
            % Default: nnet.cnn.TrainingOptionsADAM
            %
            % Returns: 
            % 
            % * _RMSE_train_: RMSE on training data.
            % * _RMSE_eval_: RMSE on validation data.
            % * _RMSE_train_N_: Normalized RMSE on training data.
            % * _RMSE_eval_N_: Normalized RMSE on validation data.
            %
            %%%
            
            arguments
                obj
                x double 
                y double 
                eval_split double {mustBeInRange(eval_split, 0, 1)} = 0;
                options {mustBeA(options, ...
                        ["nnet.cnn.TrainingOptionsADAM", ...
                        "nnet.cnn.TrainingOptionsRMSProp", ...
                        "nnet.cnn.TrainingOptionsSGDM"])} =  ...
                        trainingOptions(    'adam', ...
                                            'MaxEpochs',50,...
                                            'MiniBatchSize', 32, ...
                                            'InitialLearnRate',1e-3, ...
                                            'LearnRateSchedule', 'piecewise',...
                                            'LearnRateDropFactor', 0.7,...
                                            'LearnRateDropPeriod', 5,...
                                            'GradientDecayFactor', 0.9,...
                                            'SquaredGradientDecayFactor', 0.9,...
                                            'Verbose',true, ...
                                            'Plots','training-progress' ...
                                            );
            end
            assert(size(x,2) == obj.Architecture.num_inputs, ...
                'Size of x has to fit to obj.Architecture.num_inputs.')
            assert(size(y,2) == obj.Architecture.num_outputs, ...
                'Size of y has to fit to obj.Architecture.num_outputs.')
            assert(size(x,1) == size(y,1), ...
                'Size of x has to fit to size of y.')           
          
            % Shuffle data
            idx = randperm(size(x,1));
            x = x(idx,:);
            y = y(idx,:);
            
            % Split data in train and eval
            idx = ceil(size(x,1)*(1-eval_split));
            x_train = x(1:idx,:);
            y_train = y(1:idx,:);
            x_eval = x(idx+1:end,:);
            y_eval = y(idx+1:end,:);
            
            % Normalize data
            [x_train_N, obj.Normalization.C_in, obj.Normalization.S_in] = normalize(x_train, 'zscore');
            [y_train_N, obj.Normalization.C_out, obj.Normalization.S_out] = normalize(y_train, 'zscore');
            
            % Train network
            net = trainNetwork(x_train_N,y_train_N,obj.layers,options);
            obj.thetas = net.Layers(end-1).Weights;
            obj.Initialization.net = net;
                        
            % Eval network
            [RMSE_train, RMSE_train_N] = obj.init_eval(x_train, y_train);
            if ~isempty(x_eval)
                [RMSE_eval, RMSE_eval_N] = obj.init_eval(x_eval, y_eval);   
            else 
                RMSE_eval = NaN;
                RMSE_eval_N = NaN;
            end    
            
            % Set properties
            obj.Initialization.num_train = size(x_train,1);
            obj.Initialization.num_eval = size(x_eval,1);
            obj.Initialization.trainingoptions = options;
            obj.Initialization.RMSE_train = RMSE_train;
            obj.Initialization.RMSE_train_N = RMSE_train_N;
            obj.Info.Initialization = datetime;
            if ~isempty(x_eval)
                obj.Initialization.RMSE_eval = RMSE_eval;
                obj.Initialization.RMSE_eval_N = RMSE_eval_N;
            end
            
            % Initialize GRB - set support points
            [pos, width, depth] = obj.Buffer.bin_info();
            width = width'- 1e-12; % numerical reasons
            x_init_buffer = repmat(pos, [depth, 1]);
            rand_shift = width.*rand(size(x_init_buffer)) - width/2;
            x_init_buffer = x_init_buffer + rand_shift;
            
            % Initialize GRB - prediction on support points with initial
            % network
            x_init_buffer_N = normalize(x_init_buffer, 'center', obj.Normalization.C_in, 'scale', obj.Normalization.S_in); 
            y_init_buffer_N = predict(net, x_init_buffer_N);
            y_init_buffer = obj.re_normalize(y_init_buffer_N, obj.Normalization.S_out, obj.Normalization.C_out);
            obj.Buffer.init(x_init_buffer, y_init_buffer);  
            
            % Copy initialized GRB to allow resets
            obj.BufferInitialized = copy(obj.Buffer);
      end
      % ============================== EOF ============================== %      
      
      %%
      function [y_pred, y_pred_N] = init_pred(obj, x)
          %%%
          %   [y_pred, y_pred_N] = init_pred(obj, x)
          %%%
          %
          % Performs prediction on initially trained model.
          %
          % Arguments:
          %
          % * _x_: Inputs (rows: samples; cols: features).
          %
          % Returns: 
          %
          % * _y_pred_: Predictions on _x_.
          % * _y_pred_N_: Normalized predictions on _x_.
          %
          %%%
          
          arguments
                obj
                x double 
          end
          assert(size(x,2) == obj.Architecture.num_inputs, ...
                'Size of x has to fit to obj.Architecture.num_inputs.')
          
          x_N = normalize(x, 'center', obj.Normalization.C_in, 'scale', obj.Normalization.S_in);          
          y_pred_N = predict(obj.Initialization.net, x_N);
          y_pred = obj.re_normalize(y_pred_N, obj.Normalization.S_out, obj.Normalization.C_out);
      end
      % ============================== EOF ============================== %
      
      %% 
      function [RMSE, RMSE_N] = init_eval(obj, x, y)
          %%%
          %   RMSE = init_eval(obj, x, y)
          %%%
          %
          % Calculates the RMSE for a set of features and targets for
          % initial model. 
          %
          % Arguments:
          %
          % * _x_: Inputs (rows: samples; cols: features).
          % * _y_: Outputs (rows: samples; cols: outputs).
          % 
          % Returns:
          %
          % * _RMSE_: RMSE for initial model.
          % * _RMSE_N_: Normalized RMSE for initial model.
          %
          %%%
          
          assert(size(x,2) == obj.Architecture.num_inputs, ...
                'Size of ''x'' has to fit to ''obj.Architecture.num_inputs''.')
          assert(size(y,2) == obj.Architecture.num_outputs, ...
                'Size of ''y'' has to fit to ''obj.Architecture.num_outputs''.')
          assert(size(x,1) == size(y,1), ...
                'Size of ''x'' has to fit to size of ''y''.')
          
          % RMSE
          [y_pred, y_pred_N] = obj.init_pred(x);
          RMSE = sqrt(mean((y_pred - y).^2));
          
          % Normalized RMSE
          y_N = normalize(y, 'center', obj.Normalization.C_out, 'scale', obj.Normalization.S_out);
          RMSE_N = sqrt(mean((y_pred_N - y_N).^2));
      end
      % ============================== EOF ============================== %
      
      %% 
      function thetas = cl_update(obj, x_new, y_new, hf, deltatheta)
          %%%
          %    thetas = cl_update(obj, x_new, y_new, hf, deltatheta)
          %%%
          %
          % Updates the ANN model by changing the weights in of the last
          % layer using convex optimization.
          %
          % Arguments:
          %
          % * _x_new_: Inputs (rows: samples; cols: features).
          % * _y_new_: Outputs (rows: samples; cols: outputs).
          % * _hf_ [optional]: Fraction of additional historical used for
          % optimization. The historical data is extracted from the GRB.
          % Default: 0.1
          % * _deltatheta_ [optional]: Maximum absolute change of weights
          % per iteration. If this value is empty ([]), no restriction is
          % applied. Default: 0.1
          %
          % Returns: 
          % 
          % * _thetas_: Updated thetas.
          %
          % Note: Optimization Toolbox for lsqlin-function required.
          %
          %%%
          
          arguments
              obj
              x_new double
              y_new double
              hf (1,1) double {mustBeInRange(hf, 0, 1)} = 0.1;
              deltatheta double {mustBePositive, mustBeScalarOrEmpty} = 0.1;
          end
          assert(size(x_new,2) == obj.Architecture.num_inputs, ...
                'Size of x_new has to fit to obj.Architecture.num_inputs.')
          assert(size(y_new,2) == obj.Architecture.num_outputs, ...
                'Size of y_new has to fit to obj.Architecture.num_outputs.')
          assert(size(x_new,1) == size(y_new,1), ...
                'Size of x_new has to fit to size of y_new.')
                    
          % Find number of elements from data stream
          ds = size(x_new,1);
          
          % Get samples form GRB, ignoring the bins, which were updated by
          % the latest call of this method
          if hf~=0
            [x_buffer, y_buffer] = obj.Buffer.get(ceil(ds*hf), ...
                                                obj.latest_updated_bins);
          else
              x_buffer = [];
              y_buffer = [];
          end
          
          % Update GRB with the new samples
          obj.latest_updated_bins = obj.Buffer.add(x_new, y_new);
          
          % Bulid batch
          x = [x_new; x_buffer];
          y = [y_new; y_buffer];
          
          % Train
          [~, ~, A] = obj.cl_pred(x);
          y_N = normalize(y, 'center', obj.Normalization.C_out, 'scale', obj.Normalization.S_out); 
          options = optimoptions('lsqlin','Algorithm','interior-point','Diagnostics','off','Display','off');
          if isempty(deltatheta)
              lb = [];
              ub = [];
          else
             lb = double(obj.thetas-deltatheta); 
             ub = double(obj.thetas+deltatheta); 
          end
          thetas = lsqlin(double(A'),y_N,[],[],[],[],lb,ub,[],options);

          
          % Set properties
          obj.thetas = thetas';
          obj.Updates = [obj.Updates; {size(obj.Updates,1)+1, ds, hf, deltatheta}];
          obj.Info.LatestUpdate = datetime;
      end
      % ============================== EOF ============================== %
           
      %%
      function [y_pred, y_pred_N, A] = cl_pred(obj, x) 
          %%%
          %   [y_pred, y_pred_N, A] = cl_pred(obj, x)
          %%%
          %
          % Performs prediction on current, continually updated model.
          %
          % Arguments:
          %
          % * _x_: Inputs (rows: samples; cols: features).
          %
          % Returns: 
          %
          % * _y_pred_: Predictions on _x_.
          % * _y_pred_N_: Normalized predictions on _x_.
          % * _A_: Activations of the last hidden layer for _x_ as input.
          %
          %%%
          
          arguments
                obj
                x double 
          end
          assert(size(x,2) == obj.Architecture.num_inputs, ...
                'Size of x has to fit to obj.Architecture.num_inputs.')
          
          x_N = normalize(x, 'center', obj.Normalization.C_in, 'scale', obj.Normalization.S_in);  
          A = activations(obj.Initialization.net, x_N, obj.Initialization.net.Layers(end-2).Name);
          y_pred_N = (obj.thetas*A)';
          y_pred = obj.re_normalize(y_pred_N, obj.Normalization.S_out, obj.Normalization.C_out);
      end
      % ============================== EOF ============================== %
                  
      %%
      function [RMSE, RMSE_N] = cl_eval(obj, x, y)
          %%%
          %   [RMSE, RMSE_N] = cl_eval(obj, x, y)
          %%%
          %
          % Calculates the RMSE for a set of features and targets for
          % current, continually updated model. 
          %
          % Arguments:
          %
          % * _x_: Inputs (rows: samples; cols: features).
          % * _y_: Outputs (rows: samples; cols: outputs).
          %
          % Returns:
          %
          % * _RMSE_: RMSE for initial model.
          % * _RMSE_N_: Normalized RMSE for initial model.
          %
          %%% 
          
          assert(size(x,2) == obj.Architecture.num_inputs, ...
                'Size of ''x'' has to fit to ''obj.Architecture.num_inputs''.')
          assert(size(y,2) == obj.Architecture.num_outputs, ...
                'Size of ''y'' has to fit to ''obj.Architecture.num_outputs''.')
          assert(size(x,1) == size(y,1), ...
                'Size of ''x'' has to fit to size of ''y''.')

          % RMSE
          [y_pred, y_pred_N, ~] = obj.cl_pred(x);
          RMSE = sqrt(mean((y_pred - y).^2));
          
          % Normalized RMSE
          y_N = normalize(y, 'center', obj.Normalization.C_out, 'scale', obj.Normalization.S_out);
          RMSE_N = sqrt(mean((y_pred_N - y_N).^2));
      end
      % ============================== EOF ============================== %
            
      %% 
      function thetas = get_thetas(obj)
          %%%
          %    get_thetas(obj)
          %%%
          %
          % Returns the current thetas of the CLNN-object.
          %
          % Arguments:
          %
          % * _None_
          %
          % Returns: 
          % 
          % * _thetas_: Current thetas.
          %
          %%%
          
          thetas = obj.thetas;
      end
      % ============================== EOF ============================== %
      
      %% 
      function reset(obj)
          %%%
          %    reset(obj)
          %%%
          %
          % Resets the CLNN-object to the state after initial training.
          %
          % Arguments:
          %
          % * _None_
          %
          % Returns: 
          % 
          % * _None_
          %
          %%%
          
          obj.Buffer = copy(obj.BufferInitialized);
          obj.thetas = obj.Initialization.net.Layers(end-1).Weights;
          obj.latest_updated_bins = double.empty;
          obj.Updates = table('Size',[0, 4], ...
                        'VariableTypes', {'double', 'double', 'double', 'double'},...
                        'VariableNames',{'No', 'ds', 'hf', 'deltatheta'} ...
                        );
          obj.Info.LatestUpdate = NaN;
          obj.Info.LatestReset = datetime;
      end
      % ============================== EOF ============================== %
    
   end
   
   %% Static Methods 
   methods (Static, Access = private)
      function ret = re_normalize(N, S, C)
          %%%
          %   ret = re_normalize(N, S, C)
          %%%
          %
          % Static and private method for inverse of nomalization.
          %
          % Arguments:
          %
          % * _N_: Normalized values.
          % * _S_: Scaling values.
          % * _C_: Centering values.
          % 
          % Returns:
          %
          % * _ret_: Values.
          %
          %%%
          
          ret = N.*S+C;
      end
      % ============================== EOF ============================== %
  end
end