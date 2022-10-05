%% Extreme Learning Machine (ELM)
% Note: This implementation allows the weight change per iteration to be
% restricted.
%
%%

classdef ELM < handle    
    %% Properties 
    % * _Architecture_: Struct containing information on the ELMs
    % architecture; *SetAccess*: immutable; *GetAccess*: public
    % * _Buffer_: Gridded Ring Buffer object to manage historical data;
    % *SetAccess*: protected, *GetAccess*: public
    % * _BufferInitialized_: Copy of _Buffer_ that is created after 
    % initialization. Needed for resetting to the state after initial 
    % training; *SetAccess*: private, *GetAccess*: private  
    % * _Info_: Struct containing information about changes of the object; 
    % *SetAccess*: protected, *GetAccess*: public
    % * _Initialization_: Struct containing information about the initial
    % training; *SetAccess*: protected, *GetAccess*: public
    % * _latest_updated_bins_: Array containing indices (columns) of bins
    % (rows) that should be excluded when calling the get-method of the GRB; 
    % *SetAccess*: private, *GetAccess*: private
    % * _Normalization_: Struct containing information about the
    % normalization of the input and output data; *SetAccess*: private,
    % *GetAccess*: private 
    % * _thetas_: Vector containing the weights, which are updated during
    % the continual learning phase; *SetAccess*: private, *GetAccess*:
    % private 
    % * _Updates_: Table containing information about the model updates 
    % during the continual learning phase; *SetAccess*: protected,
    % *GetAccess*: public 
    % * _W_: Weights and biases of the hidden layer; *SetAccess*: immutable,
    % *GetAccess*: private 
    
    properties(SetAccess = immutable, GetAccess = public)
        Architecture = struct(  'num_inputs', NaN, ...
                                'num_outputs', NaN, ...
                                'num_nodes', NaN...
                                );
    end
    
    properties(SetAccess = protected, GetAccess = public)
        Buffer = GRB.empty;
        Initialization = struct('thetas', NaN, ...
                                'num_train', NaN, ...
                                'num_eval', NaN, ...
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
        W;
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
        function obj = ELM(x_limits, num_outputs, nodes, approx_grid_dx, samples_per_bin)
            %%%
            %   obj = ELM(x_limits, num_outputs, nodes, approx_grid_dx, samples_per_bin) 
            %%%
            %
            % Creates an non-trained object of ELM with empty GRB. 
            %
            % Arguments:
            %
            % * _x_limits_: 2D-array containing the lower (first column)
            % and upper (second column) boundaries of the considered input
            % space: [x1_min, x1_max; ...].
            % * _num_outputs_: Number of model outputs.
            % * _nodes_ [optional]: Nodes per layers. Default: 132
            % * _approx_grid_dx_ [optional]: Approx. grid spacing for each
            % dimension of the input space:discretization in x1, x2, ...
            % Default: diff(x_limits, [], 2)/10
            % * _samples_per_bin_ [optional]: number of samples per bin. 
            % Default: 5
            %
            % Returns: 
            %
            % * _ELM-object_
            %
            %%%
            
            arguments
                x_limits (:,2) double                                                   
                num_outputs (1,1) {mustBePositive, mustBeInteger}
                nodes (1,1) {mustBePositive, mustBeInteger} = 132;
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
            obj.Architecture.num_nodes = nodes;            
            obj.Info.Build = datetime;
            
            % Set random weights and biases for the hidden layer
            obj.W = (rand(obj.Architecture.num_inputs+1, nodes)-0.5)*2;
            
            % Setup GRB
            obj.Buffer = GRB(x_limits, num_outputs, approx_grid_dx, samples_per_bin);      
        end
      
      %% Methods
      function [RMSE_train, RMSE_eval, ...
                RMSE_train_N, RMSE_eval_N] = init_train(obj, x, y, eval_split)
            %%% 
            %   [RMSE_train, RMSE_eval, RMSE_train_N, RMSE_eval_N] =
            %   init_train(obj, x, y, eval_split) 
            %%%
            %
            % Performs initial training on static dataset. 
            %
            % Arguments: 
            %
            % * _x_: Inputs (rows: samples; cols: features).
            % * _y_: Outputs (rows: samples; cols: outputs).
            % * _eval_split_ [optional]: fraction of data separated for
            % validation. Default: 0
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
            
            % Train ELM
            H = obj.activation([x_train_N, ones(size(x_train_N,1),1)]*obj.W);
            H_inv = pinv(H);
            obj.thetas = H_inv*y_train_N;
            obj.Initialization.thetas = obj.thetas;
                        
            % Eval ELM
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
            
            % Initialize GRB - prediction on support points with init ELM
            y_init_buffer = obj.init_pred(x_init_buffer);
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
          % Performs prediction on initial model.
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
          H = obj.activation([x_N, ones(size(x_N,1),1)]*obj.W);
          y_pred_N = H*obj.thetas;
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
          % Updates the ELM.
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
          % applied. Default: [] 
          %
          % Returns: 
          % 
          % * _thetas_: Updated thetas.
          %
          %%%
          
          arguments
              obj
              x_new double
              y_new double
              hf (1,1) double {mustBeInRange(hf, 0, 1)} = 0.1;
              deltatheta  double {mustBePositive, mustBeScalarOrEmpty} = [];
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
          x_N = normalize(x, 'center', obj.Normalization.C_in, 'scale', obj.Normalization.S_in);  
          y_N = normalize(y, 'center', obj.Normalization.C_out, 'scale', obj.Normalization.S_out); 
          H = obj.activation([x_N, ones(size(x_N,1),1)]*obj.W);
          if isempty(deltatheta)
            H_inv = pinv(H);
            thetas = H_inv*y_N;
          else
              options = optimoptions('lsqlin','Algorithm','interior-point','Diagnostics','off','Display','off');
              lb = double(obj.thetas-deltatheta); 
              ub = double(obj.thetas+deltatheta); 
              thetas = lsqlin(double(H),y_N,[],[],[],[],lb,ub,[],options);
          end
          
          % Set properties
          obj.thetas = thetas;
          obj.Updates = [obj.Updates; {size(obj.Updates,1)+1, ds, hf, deltatheta}];
          obj.Info.LatestUpdate = datetime;
      end
      % ============================== EOF ============================== %
           
      %%
      function [y_pred, y_pred_N] = cl_pred(obj, x) 
          %%%
          %   [y_pred, y_pred_N, A] = cl_pred(obj, x)
          %%%
          %
          % Performs prediction on current, continual updated model.
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
          H = obj.activation([x_N, ones(size(x_N,1),1)]*obj.W);
          y_pred_N = H*obj.thetas;
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
          % current, continual updated model. 
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
          [y_pred, y_pred_N] = obj.cl_pred(x);
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
          % Returns the current thetas of the ELM-object.
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
          % Resets the ELM-object to the state after initial training.
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
          obj.thetas = obj.Initialization.thetas;
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
       
      function out = activation(in)
          %%%
          %   out = activation(in)
          %%%
          %
          % Static and private method for activation.
          %
          % Arguments:
          %
          % * _in_: Inputs.
          % 
          % Returns:
          %
          % * _out_: Inputs activated with sigmoid function.
          %
          %%%
          
          % Sigmoid-function
          out = 1./(1+exp(-in));
      end
       
       
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