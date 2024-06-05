function [id_opt, Z_opt, dis_opt, N_opt] = DPSA_DDDP_UTF8( ...
    Z_up, lb, ub, Z_ori, Z_fin, Zmaxmin, id, ...
    Q_in, Q_iflow, reservoir, aZD, aQmin, aZQ, ahmaxmin, aHK, jisuanbuchang, aZV, aOH, ah, aN, aNmin, QQmaxmin, afa)
% ��ɢ΢�ֶ�̬�滮��Discrete Differential Dynamic Program, DDDP��for DPSA
% Y2021-11-12Y
% UTF-8

%% ��ʼ������
numIte = 2;                     % ��������
n = 201;                         % ��ɢ�������ֵ����Ϊ5���ϵ�����
step_Z = (Zmaxmin(1) - Zmaxmin(2)) / (n - 1);

[m, N] = size(Z_up);            % ʱ��������ˮ������
T = m - 1;                      % ʱ������
Z_up(1, id) = Z_ori;
Z_up(m, id) = Z_fin;
Z_lim = min(ub);

Q_dis = zeros(T, n + 1, N);         % ʱ����й����
N_out = zeros(T, n + 1);            % ʱ��ƽ������
loc = zeros(T, n + 1);              % ���Ź켣ָ��
M_Z = zeros(T, 1);              % ʱ��ĩ��ɢ����
id_opt = zeros(T, 1);           % ���Ź켣���
Z_opt = zeros(T + 1, 1);      % ���Ź켣ˮλ
Z_opt(1, 1) = Z_ori;
Z_opt(T + 1, 1) = Z_fin;

%% ѭ������
numTemp = 0;                % ��ǰ��������
while true
        % ����ÿ��ʱ��
    for i = 1 : T
        % ����ʱ��ĩ��ɢ��
        Z1 = lb(i+1) : step_Z : ub(i+1);
        M_Z(i) = length(Z1);
        if numTemp == 0
            Z1(length(Z1) + 1) = Z_lim;
            M_Z(i) = M_Z(i) + 1;
        end
        if i == 1
                % ��1ʱ�Σ�����ÿ��ʱ��ĩ��ɢ��
            for j = 1 : M_Z(i)
                % �����ˮ����й������ˮ��Ⱥʱ�γ���
                Z_up(i + 1, id) = Z1(j);
                Q_dis(i, j, :) = Calculate_dis(Z_up(1:2, :), Q_in(i), Q_iflow(i, :), jisuanbuchang, aZV);
                for k = 1 : N
                    N_out(i, j) = N_out(i, j) + Constraint(Q_dis(i, j, k), Z_up(i, k), Z_up(i+1, k), reservoir(k), aZD, aQmin, aZQ, ahmaxmin, aHK, jisuanbuchang, aOH, ah, aN, aNmin, QQmaxmin(:,i),afa);
                end
                
                % ���Ź켣ָ��ָ���ʼˮλ
                loc(i, j) = 1;
            end
        else
                % ����ʱ�Σ�����ÿ��ʱ��ĩ��ɢ��
            for j = 1 : M_Z(i)
                Z_up(i + 1, id) = Z1(j);
                Z0 = lb(i) : step_Z : ub(i);
                if numTemp == 0
                    Z0(length(Z0) + 1) = Z_lim;
                end
                Q_dis2 = zeros(M_Z(i - 1), N);
                N_out2 = zeros(1, M_Z(i - 1));
                % ����ÿ��ʱ�γ���ɢ��
                for jj = 1 : M_Z(i - 1)
                    Z_up(i, id) = Z0(jj);
                    Q_dis2(jj, :) = Calculate_dis(Z_up(i : i+1, :), Q_in(i), Q_iflow(i, :), jisuanbuchang, aZV);
                    for k = 1 : N
                        N_out2(1, jj) = N_out2(1, jj) + Constraint(Q_dis2(jj, k), Z_up(i, k), Z_up(i + 1, k), reservoir(k), aZD, aQmin, aZQ, ahmaxmin, aHK, jisuanbuchang, aOH, ah, aN, aNmin, QQmaxmin(:,i),afa);
                    end
                                  
                    
                end
                                % ���Ź켣ָ��ָ���������
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
    
    % �ش�����켣
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
    
    % ���ˮ�⣨�������ʽ��ˮ��վ����������
    if id == 3
        break
    end
          
    % �жϵ�������
    numTemp = numTemp + 1;
    disp(numTemp);
    if numTemp >= numIte
        break
    else
        % ��ɢ΢���ȵ�
        if numTemp == 1
            for i = 2 : T
                if id_opt(i - 1, 1) == 1 %���ݵ�һ�����ŵĹ켣��һ���µķ�Χ,����������Ź켣��ˮλ����
                    lb(i) = Z_opt(i, 1);
                    ub(i) = Z_opt(i, 1) + 2 * step_Z;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) && ub(i) == Z_opt(i, 1)%���Ź켣��Ѷ��ˮλ������Ѵ��
                    lb(i) = Z_lim - 2 * step_Z;
                    ub(i) = Z_lim;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) && ub(i) > Z_opt(i, 1) && ub(i) - Z_opt(i, 1) < step_Z%���Ź켣��Ѷ��ˮλ�����ڷ�Ѵ�ڣ���Ѷ��ˮλ��ub����С��stepZ
                    lb(i) = ub(i) - 2 * step_Z;
                    ub(i) = ub(i);
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) && ub(i) > Z_opt(i, 1) && ub(i) - Z_opt(i, 1) >= step_Z%���Ź켣��Ѷ��ˮλ�����ڷ�Ѵ�ڣ���Ѷ��ˮλ��ub�������stepZ
                    lb(i) = Z_lim - step_Z;
                    ub(i) = Z_lim + step_Z;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) - 1 && Z_opt(i, 1) == Z_lim%���Ź켣��ˮλ���ޣ�Ѵ��
                    lb(i) = Z_lim - 2 * step_Z;
                    ub(i) = Z_lim;
                elseif id_opt(i - 1, 1) == M_Z(i - 1, 1) - 1 && Z_opt(i, 1) > Z_lim%���Ź켣��ˮλ���ޣ���Ѵ��
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
        
        % ���ó���
        N_out(:) = 0;
        
        % ���²���
        step_Z = step_Z * 2 / (n - 1);
        
    end
end

%% �ش����Ź켣
% ��й�����仯����
dis_opt = zeros(T, N);
dis_opt(T, :) = Q_dis(T, id_opt(T, 1), :);
for i = 2 : T
    dis_opt(i - 1, :) = Q_dis(i - 1, id_opt(i - 1, 1), :);
end

% �ش��ܳ���
N_opt = N_out(T, id_opt(T, 1));

end
