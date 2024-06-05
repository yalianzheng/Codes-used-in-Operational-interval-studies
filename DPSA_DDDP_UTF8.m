function [id_opt, Z_opt, dis_opt, N_opt] = DPSA_DDDP_UTF8( ...
    Z_up, lb, ub, Z_ori, Z_fin, Zmaxmin, id, ...
    Q_in, Q_iflow, reservoir, aZD, aQmin, aZQ, ahmaxmin, aHK, jisuanbuchang, aZV, aOH, ah, aN, aNmin, QQmaxmin, afa)
% 离散微分动态规划（Discrete Differential Dynamic Program, DDDP）for DPSA
% Y2021-11-12Y
% UTF-8

%% 初始化数据
numIte = 2;                     % 迭代次数
n = 201;                         % 离散点数最大值，设为5以上的奇数
step_Z = (Zmaxmin(1) - Zmaxmin(2)) / (n - 1);

[m, N] = size(Z_up);            % 时刻数量，水库数量
T = m - 1;                      % 时段数量
Z_up(1, id) = Z_ori;
Z_up(m, id) = Z_fin;
Z_lim = min(ub);

Q_dis = zeros(T, n + 1, N);         % 时段下泄流量
N_out = zeros(T, n + 1);            % 时段平均出力
loc = zeros(T, n + 1);              % 最优轨迹指针
M_Z = zeros(T, 1);              % 时段末离散点数
id_opt = zeros(T, 1);           % 最优轨迹序号
Z_opt = zeros(T + 1, 1);      % 最优轨迹水位
Z_opt(1, 1) = Z_ori;
Z_opt(T + 1, 1) = Z_fin;

%% 循环查找
numTemp = 0;                % 当前迭代次数
while true
        % 对于每个时段
    for i = 1 : T
        % 生成时段末离散点
        Z1 = lb(i+1) : step_Z : ub(i+1);
        M_Z(i) = length(Z1);
        if numTemp == 0
            Z1(length(Z1) + 1) = Z_lim;
            M_Z(i) = M_Z(i) + 1;
        end
        if i == 1
                % 第1时段，对于每个时段末离散点
            for j = 1 : M_Z(i)
                % 计算各水库下泄流量、水库群时段出力
                Z_up(i + 1, id) = Z1(j);
                Q_dis(i, j, :) = Calculate_dis(Z_up(1:2, :), Q_in(i), Q_iflow(i, :), jisuanbuchang, aZV);
                for k = 1 : N
                    N_out(i, j) = N_out(i, j) + Constraint(Q_dis(i, j, k), Z_up(i, k), Z_up(i+1, k), reservoir(k), aZD, aQmin, aZQ, ahmaxmin, aHK, jisuanbuchang, aOH, ah, aN, aNmin, QQmaxmin(:,i),afa);
                end
                
                % 最优轨迹指针指向初始水位
                loc(i, j) = 1;
            end
        else
                % 其余时段，对于每个时段末离散点
            for j = 1 : M_Z(i)
                Z_up(i + 1, id) = Z1(j);
                Z0 = lb(i) : step_Z : ub(i);
                if numTemp == 0
                    Z0(length(Z0) + 1) = Z_lim;
                end
                Q_dis2 = zeros(M_Z(i - 1), N);
                N_out2 = zeros(1, M_Z(i - 1));
                % 对于每个时段初离散点
                for jj = 1 : M_Z(i - 1)
                    Z_up(i, id) = Z0(jj);
                    Q_dis2(jj, :) = Calculate_dis(Z_up(i : i+1, :), Q_in(i), Q_iflow(i, :), jisuanbuchang, aZV);
                    for k = 1 : N
                        N_out2(1, jj) = N_out2(1, jj) + Constraint(Q_dis2(jj, k), Z_up(i, k), Z_up(i + 1, k), reservoir(k), aZD, aQmin, aZQ, ahmaxmin, aHK, jisuanbuchang, aOH, ah, aN, aNmin, QQmaxmin(:,i),afa);
                    end
                                  
                    
                end
                                % 最优轨迹指针指向最优组合
%               [N_out(i, j), loc0] = max(N_out(i - 1, 1 : M_Z(i - 1)) + N_out2(1, :));
                N_out(i, j) = max(N_out(i - 1, 1 : M_Z(i - 1)) + N_out2(1, :));
                clear weizhi wwww ww ww1
                weizhi=find(  N_out(i - 1, 1 : M_Z(i - 1)) + N_out2(1, :) == N_out(i, j)); 
                wwww=zeros(size(weizhi,2),1);
                    for ww=1:size(weizhi,2)
                        wwww(ww)=Z0(weizhi(ww));
                    end
                    [~,ww1]=max(wwww);
                    loc0=weizhi(ww1);
                Q_dis(i, j, :) = Q_dis2(loc0, :);
                loc(i, j) = loc0; 
            end
        end
    end
    
    % 回代极大轨迹
    [~, id_opt(T, 1)] = min(abs(Z1 - Z_fin));
    for i = 2 : T
        j = id_opt(T - i + 2, 1);
        id_opt(T - i + 1, 1) = loc(T - i + 2, j);
    end
    if numTemp == 0
        for i = 2 : T
            Z00 = lb(i) : step_Z : ub(i);  
            Z00(length(Z00) + 1) = Z_lim;
            Z_opt(i, 1) = Z00(id_opt(i - 1, 1));
        end
    else
        for i = 2 : T
            Z00 = lb(i) : step_Z : ub(i);  
            Z_opt(i, 1) = Z00(id_opt(i - 1, 1));
        end
    end
    
    % 检测水库（如果径流式的水电站，就跳过）
    if id == 3
        break
    end
          
    % 判断迭代次数
    numTemp = numTemp + 1;
    disp(numTemp);
    if numTemp >= numIte
        break
    else
        % 离散微分廊道
        if numTemp == 1
            for i = 2 : T
                if id_opt(i - 1, 1) == 1 %根据第一次最优的轨迹找一个新的范围,这种情况最优轨迹是水位下限
                    lb(i) = Z_opt(i, 1);
                    ub(i) = Z_opt(i, 1) + 2 * step_Z;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) && ub(i) == Z_opt(i, 1)%最优轨迹是讯限水位，且在汛期
                    lb(i) = Z_lim - 2 * step_Z;
                    ub(i) = Z_lim;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) && ub(i) > Z_opt(i, 1) && ub(i) - Z_opt(i, 1) < step_Z%最优轨迹是讯限水位，且在非汛期，且讯限水位离ub距离小于stepZ
                    lb(i) = ub(i) - 2 * step_Z;
                    ub(i) = ub(i);
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) && ub(i) > Z_opt(i, 1) && ub(i) - Z_opt(i, 1) >= step_Z%最优轨迹是讯限水位，且在非汛期，且讯限水位离ub距离大于stepZ
                    lb(i) = Z_lim - step_Z;
                    ub(i) = Z_lim + step_Z;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) - 1 && Z_opt(i, 1) == Z_lim%最优轨迹是水位上限，汛期
                    lb(i) = Z_lim - 2 * step_Z;
                    ub(i) = Z_lim;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) - 1 && Z_opt(i, 1) > Z_lim%最优轨迹是水位上限，非汛期
                    lb(i) = Z_opt(i, 1) - 2 * step_Z;
                    ub(i) = Z_opt(i, 1);
                else
                    lb(i) = Z_opt(i, 1) - step_Z;
                    ub(i) = Z_opt(i, 1) + step_Z;
                end
            end
        else
            for i = 2 : T
                if id_opt(i - 1, 1) == 1
                    lb(i) = Z_opt(i, 1);
                    ub(i) = Z_opt(i, 1) + 2 * step_Z;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1)
                    lb(i) = Z_opt(i, 1) - 2 * step_Z;
                    ub(i) = Z_opt(i, 1);
                else
                    lb(i) = Z_opt(i, 1) - step_Z;
                    ub(i) = Z_opt(i, 1) + step_Z;
                end
            end
        end
        lb(T + 1) = Z_fin - step_Z;
        ub(T + 1) = Z_fin + step_Z;
        
        % 重置出力
        N_out(:) = 0;
        
        % 更新步长
        step_Z = step_Z * 2 / (n - 1);
        
    end
end

%% 回代最优轨迹
% 下泄流量变化过程
dis_opt = zeros(T, N);
dis_opt(T, :) = Q_dis(T, id_opt(T, 1), :);
for i = 2 : T
    dis_opt(i - 1, :) = Q_dis(i - 1, id_opt(i - 1, 1), :);
end

% 回代总出力
N_opt = N_out(T, id_opt(T, 1));

end
