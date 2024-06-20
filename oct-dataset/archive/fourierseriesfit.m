function fourierseriesfit(source, save_dir, series)
    
    [~, fileName, ~] = fileparts(source);

    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    fit_img = [save_dir, '/fit_img'];
    if ~exist(fit_img, "dir")
        mkdir(fit_img);
    end

    num = readmatrix(source);
    n = size(num, 1);
    index = 1:1:n;
    % index = setdiff(1:n, index);

    % Extract x and y using the index
    x = num(index, 1)';
    y = num(index, 2)';

    % f = FourierSeries(x, y, series);
    % [f, a0, a, b] = FourierSeries(x, y, series);
    f = FourierSeries(x, y, series);

    % Plot the original data and the fitted data
    figure('Visible', 'off')
    plot(x, y, x, f);
    legend('Original Data', 'Fitted Data');
    saveas(gcf, [fit_img, '/', fileName, '.png']);
    close all;
    
    data = [x', f'];
    writematrix(data, [save_dir, '/', fileName, '.csv'], 'WriteMode', 'append');
end

function f = FourierSeries(x, y, n)
    a = zeros(1,n);
    b = zeros(1,n);
    c = zeros(length(x),n);
    s = zeros(length(x),n);
    T = max(x)-min(x);
    w = 2*pi/T;
    
    a0 = 2*trapz(x,y)/T;
    for i=1:n
        c(:,i) = diag(y'*cos(i*w*x)); % 被拟合的数据
        a(i) = 2*trapz(x,c(:,i))/T; % 求积分
        s(:,i) = diag(y'*sin(i*w*x)); % 被拟合的数据
        b(i) = 2*trapz(x,s(:,i))/T; % 求积
    end
    
    % 计算展开项，并累加
    t = x;
    M = zeros(size(t));
    for i=1:n
       M = M + a(i)*cos(i*w*t) + b(i)*sin(i*w*t);
    end
    
    % 最终结果
    f = a0/2 + M;
end