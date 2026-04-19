function ScientificComputingFramework()
    % 通用科学计算框架主函数
    clc; clear; close all;
    warning off curvefit:fittype:sethandles;  % 禁用fittype警告
    
    %% ==================== 数据输入模块 ====================
    fprintf('=== 通用科学计算框架 ===\n\n');
    
    % 1. 交互式模型选择
    model_type = questdlg('选择模型类型:', '模型选择', ...
        '二次函数(y=a*x^2+b*x+c)', '指数函数(y=a*exp(b*x))', '自定义', '二次函数(y=a*x^2+b*x+c)');
    if isempty(model_type), return; end
    
    % 2. 数据输入与验证
    [data, should_return] = getInitialData();
    if should_return, return; end
    
    %% ==================== 主计算流程 ====================
    while true
        %% ==================== 量纲分析模块 ====================
        fprintf('\n=== 量纲分析 ===\n');
        [dim_valid, dim_analysis] = performDimensionalAnalysis(model_type, data.x_dim, data.y_dim);
        
        if ~dim_valid
            choice = questdlg(sprintf('量纲不匹配! %s\n是否继续计算?', dim_analysis.message), ...
                '量纲警告', '继续', '取消', '继续');
            if strcmp(choice, '取消'), return; end
        else
            fprintf('量纲分析通过: %s\n', dim_analysis.message);
        end
        
        %% ==================== 模型配置模块 ====================
        switch model_type
            case '二次函数(y=a*x^2+b*x+c)'
                formula = 'a*x^2 + b*x + c';
                param_names = {'a','b','c'};
                initial_guess = [1 1 1];  % 参数初始值
                model_formula = @(p,x) p(1)*x.^2 + p(2)*x + p(3);  % 模型函数
                
                % 参数量纲推导
                param_dims = deriveParameterDimensions(data.x_dim, data.y_dim, formula);
                fprintf('参数量纲推导:\n');
                for i = 1:length(param_names)
                    fprintf(' - %s: [%s]\n', param_names{i}, param_dims{i});
                end
                
            case '指数函数(y=a*exp(b*x))'
                formula = 'a*exp(b*x)';
                param_names = {'a','b'};
                initial_guess = [max(data.y)/2 0.1];  % 改进的初始值估计
                model_formula = @(p,x) p(1)*exp(p(2)*x);
                
                % 指数函数特殊量纲检查
                param_dims = deriveParameterDimensions(data.x_dim, data.y_dim, formula);
                fprintf('参数量纲推导:\n');
                fprintf(' - a: [%s] (必须与y同量纲)\n', data.y_dim);
                fprintf(' - b: [%s] (指数必须无量纲)\n', param_dims{2});
                if ~strcmp(param_dims{2}, '1')
                    warndlg('指数函数的指数项b必须有量纲1 (无量纲)');
                end
                
            case '自定义'
                formula = inputdlg(['输入公式(使用标准数学函数，如a*sin(b*x)+c):\n' ...
                                  '可用函数: sin/cos/tan/exp/log/log10/sqrt等'], ...
                    '自定义模型', 1, {'a*sin(b*x)+c'});
                if isempty(formula), return; end
                formula = formula{1};
                formula = vectorizeMathExpr(formula);  % 向量化处理
                
                % 提取参数
                [param_names, initial_guess] = getParameters(formula);
                if isempty(param_names), return; end
                
                try
                    % 创建模型函数
                    model_formula = eval(['@(p,x) ' replaceSymbols(formula, param_names)]);
                    
                    % 自定义模型的量纲分析
                    param_dims = deriveParameterDimensions(data.x_dim, data.y_dim, formula);
                    fprintf('自定义模型参数量纲推导:\n');
                    for i = 1:length(param_names)
                        fprintf(' - %s: [%s]\n', param_names{i}, param_dims{i});
                    end
                catch ME
                    errordlg(sprintf('自定义公式解析失败!\n错误: %s\n请检查公式语法', ME.message));
                    return;
                end
        end
        
        %% ==================== 智能初始值估计 ====================
        initial_guess = getSmartInitialValues(param_names, initial_guess, data.x, data.y, model_type);
        if isempty(initial_guess), return; end
        
        %% ==================== 高精度最小二乘法曲线拟合 ====================
        fprintf('\n=== 高精度最小二乘法曲线拟合 ===\n');
        [fit_result, params_opt, gof] = performHighPrecisionFit(...
            model_formula, data.x, data.y, initial_guess, param_names, model_type);
        if isempty(fit_result)
            errordlg('曲线拟合失败! 请尝试不同的初始值');
            return;
        end
        
        %% ==================== 量纲一致性验证 ====================
        fprintf('\n=== 量纲一致性验证 ===\n');
        verifyDimensionalConsistency(params_opt, param_names, param_dims, data.x_dim, data.y_dim, formula);
        
        %% ==================== 高精度Q学习优化 ====================
        fprintf('\n=== 高精度Q学习优化 ===\n');
        optimal_point = refinedQLearningOptimize(fit_result, data.x, model_type);
        
        % 对二次函数验证理论极值点
        if strcmp(model_type, '二次函数(y=a*x^2+b*x+c)')
            theoretical_opt = -params_opt(2)/(2*params_opt(1));
            fprintf('理论极值点: x = %.10g\n', theoretical_opt);
            if abs(optimal_point - theoretical_opt) > 1e-6
                optimal_point = theoretical_opt;
                fprintf('采用理论极值点作为最优解\n');
            end
        end
        
        %% ==================== 结果可视化 ====================
        visualizeResults(fit_result, params_opt, data, gof, optimal_point, model_formula, model_type);
        
        %% ==================== 数据追加询问 ====================
        choice = questdlg('是否要追加数据点以提高拟合精度?', '数据追加', '是', '否', '是');
        if strcmp(choice, '否')
            break;
        end
        
        % 获取追加数据
        [new_data, should_return] = getAdditionalData(data);
        if should_return, break; end
        
        % 合并新旧数据
        data.x = [data.x, new_data.x];
        data.y = [data.y, new_data.y];
        
        % 显示更新后的数据
        fprintf('\n=== 更新后的数据集 ===\n');
        disp(array2table([data.x; data.y], 'RowNames', {'x', 'y'}));
    end
    
    fprintf('\n=== 计算完成 ===\n');
end

%% ==================== 数据获取函数 ====================
function [data, should_return] = getInitialData()
    % 获取初始数据
    should_return = false;
    while true
        input_data = inputdlg({'自变量x (空格分隔):', ...
                              '因变量y (空格分隔):', ...
                              'x单位 (如m,kg,s):', ...
                              'y单位 (如m,kg,s):'}, ...
                             '数据输入', [1 50; 1 50; 1 10; 1 10]);
        if isempty(input_data)
            should_return = true;
            data = [];
            return; 
        end
        
        try
            data.x = str2num(input_data{1});
            data.y = str2num(input_data{2});
            assert(length(data.x)>=3 && length(data.x)==length(data.y));
            
            % 存储单位信息
            data.x_dim = strtrim(input_data{3});
            data.y_dim = strtrim(input_data{4});
            if isempty(data.x_dim), data.x_dim = '1'; end  % 默认为无量纲
            if isempty(data.y_dim), data.y_dim = '1'; end
            
            break;
        catch
            errordlg('输入无效! 需满足: 1) 数值格式 2) 数据点≥3 3) x/y长度一致');
        end
    end
    
      % 显示输入数据
    fprintf('\n=== 输入数据 ===\n');
        disp(array2table([data.x; data.y], 'RowNames', {'x', 'y'}));
    fprintf('单位信息: x [%s], y [%s]\n', data.x_dim, data.y_dim);
end

function [new_data, should_return] = getAdditionalData(old_data)
    % 获取追加数据
    should_return = false;
    new_data = struct();
    
    prompt = {sprintf('追加的x值 (空格分隔，当前单位[%s]):', old_data.x_dim), ...
              sprintf('追加的y值 (空格分隔，当前单位[%s]):', old_data.y_dim)};
    
    while true
        input_data = inputdlg(prompt, '追加数据输入', [1 50; 1 50]);
        if isempty(input_data)
            should_return = true;
            return;
        end
        
        try
            new_data.x = str2num(input_data{1});
            new_data.y = str2num(input_data{2});
            assert(length(new_data.x)>=1 && length(new_data.x)==length(new_data.y));
            
            % 使用原有单位
            new_data.x_dim = old_data.x_dim;
            new_data.y_dim = old_data.y_dim;
            
            % 显示追加数据
            fprintf('\n=== 追加的数据点 ===\n');
            disp(array2table([new_data.x; new_data.y], 'RowNames', {'x_add', 'y_add'}));
            break;
        catch
            errordlg('输入无效! 需满足: 1) 数值格式 2) 数据点≥1 3) x/y长度一致');
        end
    end
end

%% ==================== 辅助函数 ====================
function formula = vectorizeMathExpr(formula)
    % 数学表达式向量化处理
    math_functions = {'sin','cos','tan','exp','log','log10','sqrt',...
                     'asin','acos','atan','sinh','cosh','tanh'};
    
    % 处理运算符向量化
    ops = {'*','/','^'};
    for op = ops
        formula = strrep(formula, op{1}, ['.' op{1}]);
    end
    
    % 处理函数向量化 (保留原函数名)
    for fn = math_functions
        formula = regexprep(formula, ['\<' fn{1} '\>\s*\('], [fn{1} '(']);
    end
end

function [params, guesses] = getParameters(formula)
    % 从公式中提取参数
    params = {};
    guesses = [];
    try
        reserved_words = {'x','i','j','pi','inf','nan',...
                         'sin','cos','tan','exp','log','log10','sqrt',...
                         'asin','acos','atan','sinh','cosh','tanh'};
        
        tokens = regexp(formula, '\<[a-zA-Z][a-zA-Z0-9]*\>', 'match');
        vars = unique(tokens);
        params = setdiff(vars, reserved_words);
        
        % 改进的初始值估计
        guesses = ones(1, length(params));
        if length(params) >= 1
            guesses(1) = 1.0;  % 第一个参数初始值
        end
        if length(params) >= 2
            guesses(2) = 0.1;  % 第二个参数通常较小
        end
      catch
        errordlg('参数提取失败! 请确认使用标准数学函数(如sin(x), 而非sind(x))');
    end
end

function initial_guess = getInitialValues(param_names, default_guess)
    % 获取参数初始值
    initial_guess = [];
    prompt = cell(1, length(param_names));
    def_values = cell(1, length(param_names));
    for i = 1:length(param_names)
        prompt{i} = sprintf('参数 %s 的初始值:', param_names{i});
        def_values{i} = num2str(default_guess(i));
    end
    input_values = inputdlg(prompt, '初始值输入', 1, def_values);
    if isempty(input_values), return; end
    
    try
        initial_guess = str2double(input_values)';
        if any(isnan(initial_guess))
            error('无效的数值输入');
        end
    catch
        errordlg('初始值必须为有效数值!');
        initial_guess = [];
    end
end

function modified_formula = replaceSymbols(formula, params)
    % 将公式中的参数符号替换为p(1),p(2)等
    modified_formula = formula;
    for i = 1:length(params)
        modified_formula = strrep(modified_formula, params{i}, sprintf('p(%d)',i));
    end
end
%% ==================== 高精度拟合函数 ====================
function [fit_result, params_opt, gof] = performHighPrecisionFit(model_func, x, y, init, param_names, model_type)
    % 高精度最小二乘法拟合
    fprintf('\n=== 高精度最小二乘法曲线拟合 ===\n');
    
    % 设置严格的优化选项
    options = optimoptions('lsqcurvefit', ...
        'Display', 'iter-detailed', ...  % 显示迭代过程
        'MaxFunctionEvaluations', 5000, ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', 1e-10, ...   % 提高到1e-10
        'OptimalityTolerance', 1e-10, ... % 提高到1e-10
        'StepTolerance', 1e-10, ...
        'FiniteDifferenceType', 'central', ... % 更精确的梯度计算
        'UseParallel', false);            % 并行计算
    
    % 参数边界约束（根据模型类型调整）
    switch model_type
        case '二次函数(y=a*x^2+b*x+c)'
            lb = [-inf, -inf, -inf]; % 宽松边界
            ub = [inf, inf, inf];
        case '指数函数(y=a*exp(b*x))'
            lb = [1e-6, -inf];       % a必须为正
            ub = [inf, inf];
        otherwise
            lb = -inf * ones(size(init));
            ub = inf * ones(size(init));
    end
    
    try
        % 多起点优化策略
        best_params = [];
        best_residual = inf;
        
        % 尝试多个初始点（特别是对指数函数）
        if contains(model_type, '指数函数')
            initial_points = [...
                init; ...                         % 原始初始值
                [max(y)/2, 0.01]; ...            % 小b值
                [max(y), 0.001]; ...             % 更小b值
                [mean(y), (max(x)-min(x))^(-1)]; % 基于数据范围的b值
            ];
            
            for i = 1:size(initial_points, 1)
                fprintf('尝试初始点 %d: %s\n', i, mat2str(initial_points(i,:), 3));
                
                [params_candidate, ~, residual, exitflag] = lsqcurvefit(...
                    model_func, initial_points(i,:), x, y, lb, ub, options);
                
                if exitflag > 0 && residual < best_residual
                    best_params = params_candidate;
                    best_residual = residual;
                    fprintf('找到更好解，残差: %.6e\n', best_residual);
                end
            end
            
            params_opt = best_params;
        else
            % 对于其他模型，使用单一优化但更严格的设置
            [params_opt, ~, residual, exitflag, output] = lsqcurvefit(...
                model_func, init, x, y, lb, ub, options);
        end
        
        if isempty(params_opt)
            error('所有初始点优化失败');
        end
        
        % 创建拟合函数
        fit_func = @(x) model_func(params_opt, x);
        
        % 计算高精度拟合优度
        y_fit = model_func(params_opt, x);
        SS_tot = sum((y - mean(y)).^2);
        SS_res = sum((y - y_fit).^2);
        rsquare = 1 - SS_res/SS_tot;
        
        % 计算调整R²考虑参数数量
        n_params = length(init);
        n_points = length(y);
        adjrsquare = 1 - (1-rsquare)*(n_points-1)/(n_points-n_params-1);
        
        % 计算RMSE和MAE
        rmse = sqrt(mean((y - y_fit).^2));
        mae = mean(abs(y - y_fit));
        
        gof = struct('rsquare', rsquare, 'adjrsquare', adjrsquare, ...
                    'rmse', rmse, 'mae', mae, 'ss_res', SS_res);
        
        fprintf('\n高精度拟合结果:\n');
        for i = 1:length(param_names)
            fprintf('%s = %.10g\n', param_names{i}, params_opt(i));
        end
        fprintf('拟合优度:\n');
        fprintf(' - R² = %.10f\n', gof.rsquare);
        fprintf(' - 调整R² = %.10f\n', gof.adjrsquare);
        fprintf(' - RMSE = %.10f\n', gof.rmse);
        fprintf(' - MAE = %.10f\n', gof.mae);
        fprintf(' - 残差平方和 = %.10e\n', gof.ss_res);
        
        fit_result = fit_func;
        
    catch ME
        errordlg(sprintf('高精度拟合失败!\n错误: %s', ME.message));
        fit_result = [];
        params_opt = [];
        gof = [];
    end
end

%% ==================== 智能初始值估计 ====================
function initial_guess = getSmartInitialValues(param_names, default_guess, x, y, model_type)
    % 智能初始值估计
    
    fprintf('智能初始值估计:\n');
    
    switch model_type
        case '二次函数(y=a*x^2+b*x+c)'
            % 使用多项式拟合获得更好的初始值
            p = polyfit(x, y, 2);
            initial_guess = [p(1), p(2), p(3)];
            
        case '指数函数(y=a*exp(b*x))'
            % 对指数函数的智能初始值估计
            y_positive = y(y > 0);
            if isempty(y_positive)
                initial_guess = default_guess;
            else
                % 取对数进行线性拟合
                log_y = log(y_positive);
                valid_x = x(y > 0);
                p = polyfit(valid_x, log_y, 1);
                initial_guess = [exp(p(2)), p(1)]; % a = exp(截距), b = 斜率
            end
            
        otherwise
            initial_guess = default_guess;
    end
    
    for i = 1:length(param_names)
        fprintf(' %s: %.10g\n', param_names{i}, initial_guess(i));
    end
end

%% ==================== 高精度Q学习优化 ====================
function optimal_x = refinedQLearningOptimize(fit_result, x_range, model_type)
    % 改进的Q学习优化算法，提高精度
    
    % 扩展搜索范围
    range_span = max(x_range) - min(x_range);
    x_min = min(x_range) - 0.1 * range_span;
    x_max = max(x_range) + 0.1 * range_span;
    
    % 根据模型类型选择优化方向
    if contains(model_type, '二次函数') 
        reward_func = @(x) -fit_result(x); % 寻找最小值
    else
        reward_func = @(x) fit_result(x);  % 寻找最大值
    end
    
    % 使用更精细的状态空间
    num_states = 500; % 增加到500个状态
    states = linspace(x_min, x_max, num_states);
    
    % 增强的Q学习参数
    alpha = 0.2;    % 学习率
    gamma = 0.95;   % 折扣因子
    epsilon = 0.3;  % 探索率
    episodes = 3000; % 增加训练轮次
    
    Q = zeros(num_states, 5);  % 更多动作: [-2,-1,0,+1,+2]
    
    for ep = 1:episodes
        state = randi(num_states);
        for step = 1:150
            % epsilon-greedy策略
            if rand < epsilon
                action = randi(5);
            else
                [~, action] = max(Q(state, :));
            end
            
            % 状态转移
            new_state = state + (action - 3); % 动作映射到步长
            new_state = max(1, min(num_states, new_state));
            
            % 奖励计算
            reward = reward_func(states(new_state));
            
            % Q值更新
            Q(state, action) = Q(state, action) + ...
                alpha * (reward + gamma * max(Q(new_state, :)) - Q(state, action));
            
            state = new_state;
        end
        
        % 动态调整探索率
        epsilon = max(0.1, epsilon * 0.999);
    end
    
    % 选择最优状态
    [~, optimal_state] = max(mean(Q, 2));
    optimal_x = states(optimal_state);
    
    % 使用fminbnd进行精细优化
    try
        if contains(model_type, '二次函数')
            optimal_x = fminbnd(fit_result, x_min, x_max, ...
                optimset('Display', 'off', 'TolX', 1e-10));
        else
            optimal_x = fminbnd(@(x) -fit_result(x), x_min, x_max, ...
                optimset('Display', 'off', 'TolX', 1e-10));
        end
    catch
        % 如果fminbnd失败，使用Q学习结果
        fprintf('使用Q学习结果作为最优解\n');
    end
    
    % 验证结果
    opt_val = fit_result(optimal_x);
    fprintf('高精度最优解:\n');
    fprintf(' - x* = %.10g\n', optimal_x);
    fprintf(' - f(x*) = %.10g\n', opt_val);
    
    % 二次验证
    test_points = linspace(x_min, x_max, 2000);
    test_values = fit_result(test_points);
    
    if contains(model_type, '二次函数')
        [min_val, min_idx] = min(test_values);
        if abs(min_val - opt_val) > 1e-8
            optimal_x = test_points(min_idx);
            fprintf('调整后最优解: x = %.10g\n', optimal_x);
        end
    else
        [max_val, max_idx] = max(test_values);
        if abs(max_val - opt_val) > 1e-8
            optimal_x = test_points(max_idx);
            fprintf('调整后最优解: x = %.10g\n', optimal_x);
        end
    end
end

%% ==================== 结果可视化 ====================
function visualizeResults(fit_result, params_opt, data, gof, optimal_point, model_formula, model_type)
    % 结果可视化
    fig = figure('Position', [100 100 1200 500], 'Name', '高精度分析结果', 'NumberTitle', 'off');

    % 子图1：数据拟合
    subplot(1,2,1);
    scatter(data.x, data.y, 'bo', 'DisplayName', '原始数据', 'LineWidth', 1.5);
    hold on;
    x_fine = linspace(min(data.x), max(data.x), 500);
    y_fine = fit_result(x_fine);
    plot(x_fine, y_fine, 'r-', 'DisplayName', '拟合曲线', 'LineWidth', 2);
    % 添加置信区间
    residuals = data.y - model_formula(params_opt, data.x);
    sigma = std(residuals);
    ci_upper = y_fine + 1.96*sigma;
    ci_lower = y_fine - 1.96*sigma;
    fill([x_fine, fliplr(x_fine)], [ci_upper, fliplr(ci_lower)], ...
        [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', '95% 置信区间');
    
    title(sprintf('高精度模型拟合 (R² = %.8f)', gof.rsquare));
    xlabel(sprintf('x [%s]', data.x_dim)); 
    ylabel(sprintf('y [%s]', data.y_dim));
    legend('Location', 'best'); 
    grid on;
    box on;
    
    % 子图2：最优解分析
    subplot(1,2,2);
    x_range = [min(data.x), max(data.x)];
    x_ext = linspace(x_range(1)-(x_range(2)-x_range(1))*0.2, ...
            x_range(2)+(x_range(2)-x_range(1))*0.2, 500);
    y_ext = fit_result(x_ext);
    
    plot(x_ext, y_ext, 'b-', 'LineWidth', 2, 'DisplayName', '拟合曲线');
    hold on;
    
    % 标记最优解
    opt_val = fit_result(optimal_point);
    plot(optimal_point, opt_val, 'ro', 'MarkerSize', 10, ...
        'MarkerFaceColor', 'r', 'DisplayName', sprintf('最优点 (%.8g, %.8g)', optimal_point, opt_val));
    
    % 添加参考线
    yl = ylim;
    line([optimal_point optimal_point], [yl(1) opt_val], ...
        'Color', 'r', 'LineStyle', '--', 'DisplayName', '最优x值');
    line([x_ext(1) optimal_point], [opt_val opt_val], ...
        'Color', 'r', 'LineStyle', '--', 'DisplayName', '最优y值');
    
    title('高精度最优解定位');
    xlabel(sprintf('x [%s]', data.x_dim)); 
    ylabel(sprintf('f(x) [%s]', data.y_dim));
    legend('Location', 'best');
    grid on;
    box on;
    
    % 添加信息标注
    annotation('textbox', [0.15 0.15 0.3 0.12], 'String', ...
        sprintf('拟合参数:\n%s = %.8g\n%s = %.8g\n%s = %.8g', ...
        'a', params_opt(1), 'b', params_opt(min(2,end)), 'c', params_opt(min(3,end))), ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'none');
    
    annotation('textbox', [0.6 0.15 0.3 0.12], 'String', ...
        sprintf('最优解:\nx = %.10g\ny = %.10g\nRMSE = %.8g\nMAE = %.8g', ...
        optimal_point, opt_val, gof.rmse, gof.mae), ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'none');
end

%% ==================== 量纲分析函数 ====================
function [is_valid, result] = performDimensionalAnalysis(model_type, x_dim, y_dim)
    % 量纲分析主函数
    result.message = '';
    is_valid = true;
    
    % 空量纲处理
    if strcmp(x_dim, '1') && strcmp(y_dim, '1')
        result.message = '无量纲数据，跳过量纲检查';
        return;
    end
    
    % 检查基本量纲格式
    if ~isValidDimension(x_dim) || ~isValidDimension(y_dim)
        result.message = '量纲格式无效，使用基本量纲组合(如m,kg,s)';
        is_valid = false;
        return;
    end
    
    % 模型特定量纲检查
    switch model_type
        case '二次函数(y=a*x^2+b*x+c)'
            % 检查x^2和x项是否量纲兼容
            x2_dim = multiplyDimensions(x_dim, x_dim);
            if ~areDimensionsCompatible(x2_dim, x_dim)
                result.message = sprintf('二次项量纲[%s]与线性项[%s]不兼容', x2_dim, x_dim);
                is_valid = false;
                return;
            end
            
            % 检查与y量纲的兼容性
            if ~areDimensionsCompatible(y_dim, x2_dim) && ~areDimensionsCompatible(y_dim, x_dim)
                result.message = sprintf('y量纲[%s]与x量纲[%s]不兼容', y_dim, x_dim);
                is_valid = false;
            else
                result.message = sprintf('二次模型量纲兼容: x[%s], y[%s]', x_dim, y_dim);
            end
            
        case '指数函数(y=a*exp(b*x))'
            % 检查指数项是否无量纲
            if ~strcmp(simplifyDimension(x_dim), '1')
                result.message = sprintf('指数函数要求x无量纲，当前x量纲[%s]', x_dim);
                is_valid = false;
            else
                % 检查a的量纲是否与y一致
                if ~areDimensionsCompatible(y_dim, '1')
                    result.message = sprintf('a的量纲[%s]必须与y[%s]一致', y_dim, y_dim);
                    is_valid = false;
                else
                    result.message = '指数模型量纲有效';
                end
            end
        otherwise
            result.message = '自定义模型量纲未验证';
    end
end

function param_dims = deriveParameterDimensions(x_dim, y_dim, formula)
    % 根据公式推导参数量纲
    param_dims = {};
    
    if contains(formula, 'a*x^2 + b*x + c')
        % 二次函数参数量纲
        param_dims{1} = divideDimensions(y_dim, multiplyDimensions(x_dim, x_dim)); % a
        param_dims{2} = divideDimensions(y_dim, x_dim); % b
        param_dims{3} = y_dim; % c
    elseif contains(formula, 'a*exp(b*x)')
        % 指数函数参数量纲
        param_dims{1} = y_dim; % a
        param_dims{2} = divideDimensions('1', x_dim); % b
    else
        % 自定义函数的简单量纲推导
        tokens = regexp(formula, '\<[a-zA-Z][a-zA-Z0-9]*\>', 'match');
        params = unique(tokens);
         params = setdiff(params, {'x'});
        
        for i = 1:length(params)
            % 简化处理：假设参数与y同量纲
            param_dims{i} = y_dim;
        end
    end
end

function verifyDimensionalConsistency(params, param_names, param_dims, x_dim, y_dim, formula)
    % 验证拟合参数的量纲一致性
    fprintf('参数量纲一致性验证:\n');
    
    for i = 1:length(params)
        param_value = params(i);
        param_dim = param_dims{i};
        
        fprintf(' - %s [%s]: 值=%.8g', param_names{i}, param_dim, param_value);
        
        % 特殊参数检查
        if strcmp(param_names{i}, 'b') && contains(formula, 'exp(b*x)')
            if ~strcmp(simplifyDimension(param_dim), '1')
                fprintf(' (警告: 指数项应有量纲1)');
            end
        end
        fprintf('\n');
    end
    
    % 验证模型输出量纲
    if ~strcmp(y_dim, '1')
        fprintf('模型输出量纲应匹配: [%s]\n', y_dim);
    end
end

%% ==================== 量纲操作工具函数 ====================
function valid = isValidDimension(dim)
    % 简单验证量纲格式
    valid = isempty(regexp(dim, '[^a-zA-Z0-9*/^]', 'once'));
end

function result = multiplyDimensions(dim1, dim2)
    % 量纲相乘
    if strcmp(dim1, '1'), result = dim2; return; end
    if strcmp(dim2, '1'), result = dim1; return; end
    result = [dim1 '*' dim2];
end

function result = divideDimensions(dim1, dim2)
    % 量纲相除
    if strcmp(dim2, '1'), result = dim1; return; end
    if strcmp(dim1, dim2), result = '1'; return; end
    result = [dim1 '/' dim2];
end

function result = powerDimensions(dim, power)
    % 量纲幂次
    if power == 0, result = '1'; return; end
    if power == 1, result = dim; return; end
    result = [dim '^' num2str(power)];
end

function compatible = areDimensionsCompatible(dim1, dim2)
    % 简化版量纲兼容性检查
        compatible = strcmp(simplifyDimension(dim1), simplifyDimension(dim2));
end

function simple_dim = simplifyDimension(dim)
    % 修复后的量纲简化函数
    if isempty(dim) || strcmp(dim, '1'), simple_dim = '1'; return; end
    
    try
        % 处理幂次 (如m^2)
        dim = regexprep(dim, '(\w+)\^(\d+)', '${repmat([$1 ''*''],1,str2num($2)-1)}$1');
        
        % 分割分子分母
        parts = strsplit(dim, '/');
        if length(parts) == 1
            num = parts{1};
            den = '1';
        else
            num = parts{1};
            den = parts{2};
        end
        
        % 分割乘法项
        num_terms = strsplit(num, '*');
        den_terms = strsplit(den, '*');
        
        % 移除空项
        num_terms(cellfun(@isempty, num_terms)) = [];
        den_terms(cellfun(@isempty, den_terms)) = [];
        
        % 排序
        num_terms = sort(num_terms);
        den_terms = sort(den_terms);
        
        % 移除相同项
        i = 1; j = 1;
        while i <= length(num_terms) && j <= length(den_terms)
            if strcmp(num_terms{i}, den_terms{j})
                num_terms(i) = [];
                den_terms(j) = [];
            elseif strcmp(num_terms{i}, den_terms{j})
                i = i + 1;
                j = j + 1;
            elseif num_terms{i} < den_terms{j}
                i = i + 1;
            else
                j = j + 1;
            end
        end
        
        % 重新构建量纲
        if isempty(num_terms) && isempty(den_terms)
            simple_dim = '1';
        elseif isempty(den_terms)
            simple_dim = strjoin(unique(num_terms), '*');
        else
            simple_dim = [strjoin(unique(num_terms), '*') '/' strjoin(unique(den_terms), '*')];
        end
        
        % 处理特殊情况
        if strcmp(simple_dim, '1/1'), simple_dim = '1'; end
        if endsWith(simple_dim, '*'), simple_dim = simple_dim(1:end-1); end
        if endsWith(simple_dim, '/'), simple_dim = [simple_dim '1']; end
        
    catch
        simple_dim = dim;  % 出错时返回原始量纲
    end
end