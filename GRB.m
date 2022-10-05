%% Gridded Ring Buffer (GRB)
% GRB class for temporary storage of samples, based on a discretized
% feature space. 
% Use this class to:
%
% * temporarily store samples with n-dimensional inputs and m-dimensional
%   outputs
% * continuously update the stored samples by a data stream according to
%   the first-in-first-out (fifo) principle within discrete regions of the
%   input space
% * get a set of samples randomly distributed over the input space
%
%%

classdef GRB < matlab.mixin.Copyable
    %% Properties
    % * _bins_depth_: Number samples to be stored per bin; *SetAccess*:
    % immutable; *GetAccess*: public 
    % * _bins_per_dim_: Number of bins along each axis of the discretized
    % feature space; *SetAccess*: immutable; *GetAccess*: public
    % * _eps_: Small number for numerical use; *SetAccess*: Constant;
    % *GetAccess*: private
    % * _grid_bounds_: 2D-array containing the lower (first column) and
    % upper (second column) boundaries of the considered input space;
    % *SetAccess*: immutable; *GetAccess*: public
    % * _grid_buffer_: Array containing all stored samples; *SetAccess*:
    % private; *GetAccess*: private 
    % * _grid_spacing_: Vector containing the grid spacing for each
    % dimension of the discretized input space; *SetAccess*: immutable;
    % *GetAccess*: public 
    % * _is_initialized_: Boolean, indicating whether the GRB is fully
    % initialized; *SetAccess*: private; *GetAccess*: private 
    % * _latest_updated_bins_: Array containing indices (columns) of
    % latest updated bins (rows); *SetAccess*: private; *GetAccess*:
    % private
    % * _next_idx_in_bins_: Index of sample per bin, which is replaced next
    % (fifo); *SetAccess*: private; *GetAccess*: private 
    % * _x_num_: Number of inputs per sample / size of input space;
    % *SetAccess*: immutable; *GetAccess*: public 
    % * _y_num_: Number of outputs per sample / size of output space;
    % *SetAccess*: immutable; *GetAccess*: public 
 
    properties(SetAccess = immutable, GetAccess = public)
        grid_bounds;
        grid_spacing;
        bins_per_dim;
        bins_depth;
        x_num;
        y_num;
    end
   
    properties(SetAccess = private, GetAccess = private)
        grid_buffer;
        next_idx_in_bins;
        latest_updated_bins;
        is_initialized = false;
    end 
    
    properties(Constant, GetAccess = private)
        eps = 1e-13;
    end
   
    methods  
        %% Constructor
        function obj = GRB(grid_bounds, y_num, approx_grid_spacing, bins_depth)
            %%%
            %   obj = GRB(grid_bounds, y_num, approx_grid_spacing, bins_depth)
            %%%
            % Creates an empty object of GRB.
            %
            % Arguments:
            %
            % * _grid_bounds_: 2D-array containing the lower (first column)
            % and upper (second column) boundaries of the considered input
            % space: [x1_min, x1_max; ...]. 
            % * _y_num_: Number of outputs per sample / size of output space. 
            % * _approx_grid_spacing_ [optional]: Approx. grid spacing for
            % each dimension of the discretized input space: discretization
            % in x1, x2, ... Default: diff(grid_bounds, [], 2)/10 
            % * _bins_depth_ [optional]: Number samples to be stored per
            % bin. Default: 1
            % 
            % Returns:
            % 
            % * _GRB-object_
            %
            %%%
            
            arguments
                grid_bounds (:,2) double
                y_num (1,1) {mustBePositive, mustBeInteger}
                approx_grid_spacing (:,1) double {mustBePositive} = diff(grid_bounds, [], 2)/10;
                bins_depth (1,1) {mustBePositive, mustBeInteger} = 1;                                   
            end
            assert(all(grid_bounds(:,1)<grid_bounds(:,2), 'all'), ...
                'Lower bounds (first column) in ''grid_bounds'' must be lower than upper limits (second column)!')
            if sum(approx_grid_spacing>(diff(grid_bounds, [], 2)/2))>0
                warning("Check discretization: At least one input-dimension has one bin only.")
            end
            
            % Calculate grid_spacing
            grid_size = diff(grid_bounds,1,2); % length of each axis
            bins_per_dim = round(grid_size./approx_grid_spacing); % number of bins per dimension
            grid_spacing = grid_size./bins_per_dim;

            % Set properties
            obj.grid_bounds = grid_bounds;
            obj.grid_spacing = grid_spacing;
            obj.bins_per_dim = bins_per_dim;
            obj.bins_depth = bins_depth; 
            obj.x_num = numel(obj.grid_spacing);
            obj.y_num = double(y_num);

            % Setup buffer
            obj.grid_buffer = NaN([ obj.bins_per_dim', ...  % discretization of input space (bins)      -> one dimension per input 
                                    obj.bins_depth, ...     % samples per bin                           -> one additional dimension
                                    obj.x_num+obj.y_num]);  % values (inputs and outputs) per sample    -> one additional dimension
                                
            % Initialization of idx along bins_depth-axis of GRB per bin 
            obj.next_idx_in_bins = ones(obj.bins_per_dim');    
        end
        
        %% Methods
        function init(obj, x, y)
            %%%
            %   init(obj, x, y)
            %%%
            % 
            % Initializes the GRB with given samples.
            %
            % Arguments:
            % 
            % * _x_: Inputs (rows: samples; cols: features).
            % * _y_: Outputs (rows: samples; cols: outputs).
            %
            % Returns: 
            % 
            % * _None_
            %
            %%%
            
            
            % Argument validation is included in obj.add.
            obj.add(x,y);
            
            % Check whether fully intialized
            missing_samples = nnz(isnan(obj.grid_buffer));
            
            if missing_samples == 0
                obj.is_initialized = true;
                obj.latest_updated_bins = {};
            else
                warning('GRB is initialized incompletely. %i sample(s) missing.', missing_samples);
            end
        end
        % ============================= EOF ============================= %
        
        %% 
        function updated_bins = add(obj, x, y)
            %%%
            %   updated_bins = add(obj, x, y)
            %%%
            %
            % Adds new samples to the buffer and overwrites older ones
            % (fifo). 
            %
            % Arguments:
            % 
            % * _x_: Inputs (rows: samples; cols: features).
            % * _y_: Outputs (rows: samples; cols: outputs).
            %
            % Returns: 
            % 
            % * _updated_bins_: Array containing indices (columns) of
            % updated bins (rows). 
            %
            %%%
            
            arguments
                obj; 
                x double;
                y double;
            end
            
            % Check dimensions of arguments            
            assert(size(x,2) == obj.x_num, ...
                sprintf('''x'' must have %i columns!', obj.x_num));
            assert(size(y,2) == obj.y_num, ...
                sprintf('''y'' must have %i columns!', obj.y_num));
            assert(size(x,1) == size(y,1), ...
                '''x'' and ''y'' must contain the same number of samples (rows)!');
                        
            % Handle inputs, which are out of range
            del = or(   any(x-obj.grid_bounds(:,1)'<0,2), ...  % check lower bound
                        any(x-obj.grid_bounds(:,2)'>0,2));     % check upper bound
                                   
            num_del = nnz(del);
            if num_del > 0
                warning("%i sample(s) out of range and therefore removed.", num_del)
                x = x(~del, :);
                y = y(~del, :);
            end
            
            % Preallocation
            updated_bins = cell(size(x));
            
            % Iterate on samples
            for ii = 1:size(x,1)
                                
                % Find bin for current sample based on lower bounds
                distance_to_lower_bounds = x(ii,:) - obj.grid_bounds(:,1)';
                num_of_dx_intervals = distance_to_lower_bounds./obj.grid_spacing';
                sample_bin = floor(num_of_dx_intervals + obj.eps) + 1;

                % Handle upper bound (case: sample on upper bound causes bin out of range)
                correction_idx = sample_bin > obj.bins_per_dim';
                sample_bin(correction_idx) = sample_bin(correction_idx)-1;

                % Check whether current bin is valid
                assert(all( (sample_bin>0) & ...                    % bin has positiv indices in all dimension
                            (sample_bin<=obj.bins_per_dim'), ...    % indices do not exceed the max. number of bins per dimension
                            'all'), ...
                            'Calculated invalid bin.');
                
                % Add current sample to buffer
                sample_bin = num2cell(sample_bin);         
                obj.grid_buffer(sample_bin{:}, ...                      % bin
                                obj.next_idx_in_bins(sample_bin{:}),... % sample in bin 
                                :) = [x(ii,:), y(ii, :)];               % all inputs and outputs
                
                % Update current index in current bin (fifo), which is limited to obj.bins_depth
                obj.next_idx_in_bins(sample_bin{:}) = mod(obj.next_idx_in_bins(sample_bin{:}),obj.bins_depth) + 1;
                
                % Add to list of updated bins
                updated_bins(ii,:) = sample_bin;
            end
            
            % Update latest_updated_bins
            updated_bins = unique(cell2mat(updated_bins),'rows');
            obj.latest_updated_bins = updated_bins;
        end
        % ============================= EOF ============================= %
      
        %%
        function [ret_x, ret_y] = get(obj, num_samples, exclude_bins)
            %%%
            %   [ret_x, ret_y] = get(obj, num_samples, exclude_bins)
            %%%
            %
            % Returns a specified number of samples (inputs and
            % corresponding outputs), which are equally distributed over
            % the discretized input space. Specified bins can be excluded. 
            %
            % Arguments:
            %
            % * _num_samples_: Number of samples that should be returned.
            % * _exclude_bins_ [optional]: Array containing indices
            % (columns) of bins (rows) that shold be excluded. Default:
            % double.empty
            %
            % Returns:
            % 
            % * _ret_x_: Inputs (rows: samples; cols: features).
            % * _ret_y_: Outputs (rows: samples; cols: outputs).
            %
            %%%
            
            arguments
                obj;
                num_samples (1,1) {mustBePositive, mustBeInteger};
                exclude_bins {mustBePositive, mustBeInteger};                
            end     
            num_valid_bins = prod(obj.bins_per_dim, 'all')-size(exclude_bins,1);            
            assert(num_samples <= num_valid_bins*obj.bins_depth, ...
                'The number of requested samples exceeds the number of available samples. Reduce ''num_samples'' or ''exclude_bins''.')
            assert(or (size(exclude_bins, 2) == obj.x_num, ...   % 1) Dimension of exclude_bins fits
                       and(size(exclude_bins, 1) == 0, ...       % 2) exlude_bins not defined -> size = [0, 0]
                           size(exclude_bins ,2) == 0)), ...     %    ...
                sprintf('The input space is discretized in %i dimensions. The number of columns of ''exclude_bins'' must fit.', obj.x_num))
            
            % Check whether GRB is completely initialized
            if ~obj.is_initialized 
                error('GRB is initialized incompletely. Complete initialization first.')
            end
          
            % Number of bins, bins to be excluded and bins to be use
            num_bins = prod(obj.bins_per_dim);
            num_bins_exclude = size(exclude_bins,1);
            num_bins_valid = num_bins-num_bins_exclude;            
            
            % Lin. indices for exclude_bins, if specified
            if ~isempty(exclude_bins)
                exclude_bins = num2cell(exclude_bins, 1);
                exclude_bins = sub2ind(obj.bins_per_dim, exclude_bins{:});
            end
            
            % Sample bins: integer mask for all bins with lin. indices
            rand_bin_mask = randperm(num_bins); 
            
            % Exclude the specified bins and adjust the random numbers
            rand_bin_mask(exclude_bins) = NaN; 
            [~, sort_idx] = sort(rand_bin_mask);
            rand_bin_mask(sort_idx) = [1:num_bins_valid, NaN(1, num_bins_exclude)];
            
            % Indexing for bins that should be used
            use_bins = find(rand_bin_mask<=num_samples);
            
            % Number of bins that are used
            num_use_bins = numel(use_bins);
          
            % Calculate the needed depth (how many samples must be taken from each bin?)
            depth = ceil(num_samples/num_use_bins); 
            
            % Rand mask for depth (idx in bins), zero-based!
            rand_depth_mask = randi(obj.bins_depth, [1, num_use_bins])-1;
            
            % Preallocation
            ret_idx = zeros(1,num_use_bins*depth); 
            
            % Iterate on depth to get multiples of valid_bins
            for ii = 1:depth
                % Store the indices in one vector
                ret_idx((ii-1)*num_use_bins+1:ii*num_use_bins) = use_bins + rand_depth_mask*num_bins;  
                % Update the random depth (zero based!)
                rand_depth_mask = mod(rand_depth_mask+1, obj.bins_depth);
            end
            
            % Discard excess samples (necessary, because multiple of num_use_bins)
            ret_idx = ret_idx(1:num_samples);
                       
            % Get all x-values
            ret_x = zeros(num_samples, obj.x_num); % Preallocate
            for ii = 1:obj.x_num
                idx = ret_idx + (ii-1)*num_bins*obj.bins_depth; % Lin. indices!
                ret_x(:, ii) = obj.grid_buffer(idx);
            end
            
            % Get all y-values
            ret_y = zeros(num_samples, obj.y_num); % Preallocate
            for ii = 1:obj.y_num
                idx = ret_idx + (ii-1+obj.x_num)*num_bins*obj.bins_depth; % Lin. indices!
                ret_y(:, ii) = obj.grid_buffer(idx); 
            end
        end
        % ============================= EOF ============================= % 
        
        %%
        function [pos, width, depth] = bin_info(obj)
            %%%
            %   [pos, width, depth] = bin_info(obj)
            %%%
            %
            % Calculates and returns the centers of all bins.
            %
            % Arguments
            % 
            % * _None_
            %
            % Returns
            %
            % * _pos_: Array containing positons of centers (columns) for
            % all bins (rows).
            % * _width_: Size of bins per dimension.
            % * _depth_: Number samples to be stored per bin.
            %
            %%%
            
            % Preallocation
            pos = cell(1, obj.x_num);
            
            % Iterate on input dimensions and find centers of bins per
            % dimension
            for ii = 1:obj.x_num
                pos{ii} =    obj.grid_bounds(ii,1)+(obj.grid_spacing(ii)/2): ...
                             obj.grid_spacing(ii): ...
                             obj.grid_bounds(ii,2);
            end
            
            % Create grid and reshape to value pairs
            [pos{:}] = ndgrid(pos{:}); 
            for ii = 1:obj.x_num 
                pos{ii} = reshape(pos{ii},[],1);
            end 
            
            % Convert cell to array
            pos = [pos{:}];
            
            % Get grid_spacing
            width = obj.grid_spacing;
            
            % Get number of samples per bin
            depth = obj.bins_depth;                       
        end
        % ============================= EOF ============================= %
        
   end

end