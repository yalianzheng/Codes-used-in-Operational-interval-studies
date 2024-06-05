tic
clc
clc,clear,close all



%% ��ȡ����
DATA1=xlsread('DATA',1); %��������Լ�����������ѧϰģ�͵ó������±߽�
DATA2=xlsread('DATA',2); %������������������������
DATA3=xlsread('DATA',3); %��ʼˮλ��ĩˮλ
DATA4=xlsread('DATA',4); %ˮλ��������

DATA_su=xlsread('DATA_su',1); %��������Լ�����������ѧϰģ�͵ó������±߽�

jisuanbuchang = 24;
Zmaxmin = [400 200 78.5;350 160 78.5]; %ˮλ������
lb = zeros(3,T+1);   ub = zeros(3,T+1); %���ռ�
jx = zeros(T,3);
BB = zeros(T,2);

%% �����������ž���
for afa = afa_de
    %% �������ľ���
    
    temp1 = [];
    temp2 = [];
    temp3 = [];
    temp4 = [];
    temp5 = [];
    temp6 = [];
    temp7 = [];
    temp6_1 = [];
    temp7_1 = [];
    temp8 = [];
    temp9 = [];
    temp10 = [];
    temp11 = [];
    temp12 = [];
    temp13 = [];
    temp14 = [];
    temp15 = [];
    temp16 = [];
    temp17 = [];
    temp18 = [];
    temp19 = [];
    
    tempR1 = [];
    tempR2 = [];
    tempR3 = [];
    tempR4 = [];
    tempR5 = [];
    tempR6 = [];
    tempR7 = [];
    tempR8 = [];
    tempR9 = [];
    tempR10 = [];
    tempR11 = [];
    tempR12 = [];
    tempR13 = [];
    tempR14 =[];
    tempR15 = [];
    tempR16 = [];
    tempR17 = [];
    tempR18 = [];
    tempR19 = [];
 
    for round = round_1:1:round_2 %size���ص���������i��ѭ�����ٴε���˼  round=1:(size(Q_in, 1)-T)
        
        clearvars -except reservoir n  aOH aNmin ah aN aZD aQmin aZQ ahmaxmin aHK aZV DATA1 DATA2 DATA3 DATA4 T jisuanbuchang Zmaxmin lb ub  DATA_su BB jx round round_1 round_2 afa_de afa temp1 temp2  temp3 temp4 temp5 temp6 temp7 temp6_1 temp7_1 temp8 temp9 temp10 temp11 temp12 temp13 temp14 temp15 temp16 temp17 temp18 temp19 tempR1 tempR2 tempR3 tempR4 tempR5 tempR6 tempR7 tempR8 tempR9 tempR10 tempR11 tempR12 tempR13 tempR14 tempR15 tempR16 tempR17 tempR18 tempR19 
         
       %% ��ʼ����
        Q_in_round = DATA2(round:round+T-1,1); % ���ε�����T*1�����sby����
        Q_iflow_round = DATA2(round:round+T-1,2:3); % ���ε�����T*1�����sby-ghy��ghy-gbz����
        Z_ori = [DATA3(round,1) DATA3(round,2) 78.5 ]; %��ʼˮλ - ��ʵ��ˮλѡȡ
        Z_fin = [DATA3(round+T,1) DATA3(round+T,2) 78.5 ];  %ĩˮλ - ������ʵ��ˮλ��ƽ��ˮλѡȡ
        
        %������ʼ�⣬Ӱ�첻��
        Z_up(1:T+1,1) = DATA3(round,1);
        Z_up(1:T+1,2) = DATA3(round,2);
        Z_up(1:T+1,3) = Z_ori(3);
        
        %��������
        
        round_number1 =  zeros(T,1);
        for i=1:length(round_number1)
            round_number1(i) = round;
        end
        round_number2 =  zeros(T+1,1);
        for i=1:length(round_number2)
            round_number2(i) = round;
        end
        round_number3 =  zeros(T-1,1);
        for i=1:length(round_number3)
            round_number3(i) = round;
        end
        %60�������ˮλ��������
        for i = 1:T+1
            ub(1,i) = DATA4(round+i-1,1);
            ub(2,i) = DATA4(round+i-1,2);
            ub(3,1:T+1) = 78.5;
        end
        lb(1,1:T+1) = 350;lb(2,1:T+1) = 160; lb(3,1:T+1) = 78.5;
        %ub(1,1:T+1) = uub(round:(round+T),1)';ub(2,1:T+1) = uub(round:(round+T),2)'; ub(3,1:T+1) = 78.5;
        %lb(1,1:T+1) = llb(round:(round+T),1)';lb(2,1:T+1) = llb(round:(round+T),2)'; lb(3,1:T+1) = 78.5;
        
        %ˮ����-�����ҵĳ����ϱ߽�
        QQmaxmin(1,:)=(DATA1(round:round+T-1, 9))'; %ˮ�����Ͻ�-ת��
        QQmaxmin(2,:)=(DATA1(round:round+T-1, 1))'; %ˮ�����½�-ת��
        QQmaxmin(3,:)=(DATA1(round:round+T-1, 25))'; %�������Ͻ�-ת��
        QQmaxmin(4,:)=(DATA1(round:round+T-1, 17))'; %�������½�-ת��
        
        %���帲��������,ˮ����-�����ҵ����±߽�
        CHULi_max(:,1)=(DATA_su(round:round+T-1, 1)); %ˮ��������Ͻ�-ת��
        CHULi_max(:,2)=(DATA_su(round:round+T-1, 2)); %�����ҳ����Ͻ�-ת��
        CHULi_max(:,3)=(DATA_su(round:round+T-1, 3)); %�������Ͻ�-ת��
        
        CHULi_REQUIRE(:,1)=(DATA_su(round:round+T-1, 4)); %ˮ��������Ͻ�-ת��
        
        %% ����round
        IT = 10;   % iteration times(��������)
        
        % DPSA
        res_Z = zeros(IT,n,T+1);
        N_total = zeros(IT,n);
        id_opt = zeros(IT,n,T);
        dis_opt = zeros(IT,n,T,n);
        N_opt = zeros(IT,n);
        Q_up = zeros(T,n);
        for i = 1:IT
            for j = 1:n
                id = j;
                [id_opt(i,j,:),res_Z(i,j,:),dis_opt(i,j,:,:),N_opt(i,j)] = DPSA_DDDP_UTF8(Z_up,lb(j,:),ub(j,:),Z_ori(j),Z_fin(j),Zmaxmin(:,j),id,Q_in_round ,Q_iflow_round,reservoir,aZD,aQmin,aZQ,ahmaxmin,aHK,jisuanbuchang,aZV,aOH,ah,aN,aNmin,QQmaxmin,afa);
                Z_up(:,j) = res_Z(i,j,:);
                Q_up(:,:) = dis_opt(i,j,:,:);
            end
        end
        
        
        %% ��ȡ������+�ͷ��������
        AAA1=1;AAA2=1;AAA3=1;AAA4=1;   %�ͷ��������
        
        [m,~] = size(Z_up);
        
        T = m - 1;
        N_out_sby = zeros(T,1);CHULI_out_sby = zeros(T,1);
        N_out_ghy = zeros(T,1);CHULI_out_ghy = zeros(T,1);
        N_out_gbz = zeros(T,1);CHULI_out_gbz = zeros(T,1);
        
        for i=1:T
            ZD = find_data("shuibuya",'ZD');
            Z1=Z_up(i,1);Z2=Z_up(i+1,1);
            Qmax = Inter(ZD,(Z1+Z2)/2,'WL_DC','linear');
            Qmin = find_data("shuibuya",'Qmin');
            if Q_up(i,1) <= Qmax && Q_up(i,1) >= Qmin
                ZQ = find_data("shuibuya",'ZQ');
                Z_down = Inter(ZQ,Q_up(i,1),'WL_D','linear');
                OH = 1.3;
                h = (Z1 + Z2)/2 - Z_down - OH;
                hmaxmin = find_data("shuibuya",'Head');
                if h <= hmaxmin(1) && h >= hmaxmin(2)
                    HK = find_data("shuibuya",'K');
                    K = Inter(HK,h,'WL_S','linear');
                    N_out_sby(i,1) = K*h*Q_up(i,1)*jisuanbuchang/10^8;
                    CHULI_out_sby (i,1) = K*h*Q_up(i,1)/1000;
                    
                    [Nmax,Nmin] = find_data3(h,"shuibuya");
                    if N_out_sby(i,1) > Nmax*1000*jisuanbuchang/10^8
                        N_out_sby(i,1) = Nmax*1000*jisuanbuchang/10^8;
                    elseif N_out_sby(i,1) < Nmin*1000*jisuanbuchang/10^8
                        AAA1 = 1+AAA1;
                    end
                else
                    N_out_sby(i,1) = -10000;
                    AAA2 = 1+AAA2;
                end
            elseif  Q_up(i,1) > Qmax
                N_out_sby(i,1) = -5000;
                AAA3 = 1+AAA3;
            else
                N_out_sby(i,1) = -10^8;
                AAA4 = 1+AAA4;
            end
        end
        
        for i=1:T
            ZD = find_data("geheyan",'ZD');
            Z1=Z_up(i,2);Z2=Z_up(i+1,2);
            Qmax = Inter(ZD,(Z1+Z2)/2,'WL_DC','linear');
            Qmin = find_data("geheyan",'Qmin');
            if Q_up(i,2) <= Qmax && Q_up(i,2) >= Qmin
                ZQ = find_data("geheyan",'ZQ');
                Z_down = Inter(ZQ,Q_up(i,2),'WL_D','linear');
                OH = 0.6;
                h = (Z1 + Z2)/2 - Z_down - OH;
                hmaxmin = find_data("geheyan",'Head');
                if h <= hmaxmin(1) && h >= hmaxmin(2)
                    HK = find_data("geheyan",'K');
                    K = Inter(HK,h,'WL_S','linear');
                    N_out_ghy(i,1) = K*h*Q_up(i,2)*jisuanbuchang/10^8;
                    CHULI_out_ghy (i,1) = K*h*Q_up(i,2)/1000;
                    [Nmax,Nmin] = find_data3(h,"geheyan");
                    if N_out_ghy(i,1) > Nmax*1000*jisuanbuchang/10^8
                        N_out_ghy(i,1) = Nmax*1000*jisuanbuchang/10^8;
                    elseif N_out_ghy(i,1) < Nmin*1000*jisuanbuchang/10^8
                        AAA1=1+AAA1;
                    end
                else
                    N_out_ghy(i,1) = -10000;
                    AAA2=1+AAA2;
                end
            elseif  Q_up(i,2) > Qmax
                N_out_ghy(i,1) = -5000;
                AAA3=1+AAA3;
            else
                N_out_ghy(i,1) = -10^8;
                AAA4=1+AAA4;
            end
        end
        
        for i=1:T
            ZD = find_data("gaobazhou",'ZD');
            Z1=Z_up(i,3);Z2=Z_up(i+1,3);
            Qmax = Inter(ZD,(Z1+Z2)/2,'WL_DC','linear');
            Qmin = find_data("gaobazhou",'Qmin');
            if Q_up(i,3) <= Qmax && Q_up(i,3) >= Qmin
                ZQ = find_data("gaobazhou",'ZQ');
                Z_down = Inter(ZQ,Q_up(i,3),'WL_D','linear');
                h = (Z1 + Z2)/2 - Z_down - 0.3;
                hmaxmin = find_data("gaobazhou",'Head');
                if h <= hmaxmin(1) && h >= hmaxmin(2)
                    HK = find_data("gaobazhou",'K');
                    K = Inter(HK,h,'WL_S','linear');
                    N_out_gbz(i,1) = K*h*Q_up(i,3)*jisuanbuchang/10^8;
                    CHULI_out_gbz (i,1) = K*h*Q_up(i,3)/1000;
                    [Nmax,Nmin] = find_data3(h,"gaobazhou");
                    if N_out_gbz(i,1) > Nmax*1000*jisuanbuchang/10^8
                        N_out_gbz(i,1) = Nmax*1000*jisuanbuchang/10^8;
                    elseif N_out_gbz(i,1) < Nmin*1000*jisuanbuchang/10^8
                        AAA1=1+AAA1;
                    end
                else
                    N_out_gbz(i,1) = -10000;
                    AAA2=1+AAA2;
                end
            elseif  Q_up(i,3) > Qmax
                N_out_gbz(i,1) = -5000;
                AAA3=1+AAA3;
            else
                N_out_gbz(i,1) = -10^8;
                AAA4=1+AAA4;
            end
        end
        
        AAA = [AAA1; AAA2; AAA3; AAA4];
        N_sep = [ N_out_sby, N_out_ghy,  N_out_gbz];
        
        %%  ������ˮ
        N_out_sby2 = zeros(T,1);
        N_out_ghy2 = zeros(T,1);
        N_out_gbz2 = zeros(T,1);
        for i=1:T
            Z1=Z_up(i,1);Z2=Z_up(i+1,1);
            ZQ = find_data("shuibuya",'ZQ');
            Z_down = Inter(ZQ,Q_up(i,1),'WL_D','linear');
            OH = 1.3;
            h = (Z1 + Z2)/2 - Z_down - OH;
            HK = find_data("shuibuya",'K');
            K = Inter(HK,h,'WL_S','linear');
            N_out_sby2(i,1) = K*h*Q_up(i,1)*jisuanbuchang/10^8;
            if N_out_sby2(i,1)> N_out_sby(i,1)
                Qwaste1(i,1)=Q_up(i,1)-N_out_sby(i,1)*10^8/(K*h*jisuanbuchang);
            else
                Qwaste1(i,1)=0;
            end
        end
        
        for i=1:T
            Z1=Z_up(i,2);Z2=Z_up(i+1,2);
            ZQ = find_data("geheyan",'ZQ');
            Z_down = Inter(ZQ,Q_up(i,2),'WL_D','linear');
            OH = 0.6;
            h = (Z1 + Z2)/2 - Z_down - OH;
            HK = find_data("geheyan",'K');
            K = Inter(HK,h,'WL_S','linear');
            N_out_ghy2(i,1) = K*h*Q_up(i,2)*jisuanbuchang/10^8;
            if N_out_ghy2(i,1)> N_out_ghy(i,1)
                Qwaste1(i,2)=Q_up(i,2)-N_out_ghy(i,1)*10^8/(K*h*jisuanbuchang);
            else
                Qwaste1(i,2)=0;
            end
        end
        
        for i=1:T
            Z1=Z_up(i,3);Z2=Z_up(i+1,3);
            ZQ = find_data("gaobazhou",'ZQ');
            Z_down = Inter(ZQ,Q_up(i,3),'WL_D','linear');
            h = (Z1 + Z2)/2 - Z_down - 0.3;
            HK = find_data("gaobazhou",'K');
            K = Inter(HK,h,'WL_S','linear');
            N_out_gbz2(i,1) = K*h*Q_up(i,3)*jisuanbuchang/10^8;
            if N_out_gbz2(i,1)> N_out_gbz(i,1)
                Qwaste1(i,3)=Q_up(i,3)-N_out_gbz(i,1)*10^8/(K*h*jisuanbuchang);
            else
                Qwaste1(i,3)=0;
            end
        end
        
        for i=1:T
            QW_waste(i,1)=Qwaste1(i,1)*jisuanbuchang*3600/10^4;
            QW_waste(i,2)=Qwaste1(i,2)*jisuanbuchang*3600/10^4;
            QW_waste(i,3)=Qwaste1(i,3)*jisuanbuchang*3600/10^4;
        end
        
        QW_QJ = sum(QW_waste,2);
        QW_FinSum = sum(QW_waste);
        QW_FinSum_all = sum(QW_FinSum);
        
        N_FinSum = sum(N_out_sby+N_out_ghy+N_out_gbz);
        N_out_jieguo(:,1)=N_out_sby;N_out_jieguo(:,2)=N_out_ghy;N_out_jieguo(:,3)=N_out_gbz;
        
        CHULI_out =  [CHULI_out_sby, CHULI_out_ghy ,CHULI_out_gbz];
        
        %% ���㸲����
        
        bbb1=0; bbb2=0;
        for j=1:2
            for i=1:T
                if Q_up(i,j)>QQmaxmin(j*2,i) && Q_up(i,j)<QQmaxmin(j*2-1,i)
                    if j==1
                        bbb1=bbb1+1;
                        BB(i,j)=1;
                    else
                        bbb2=bbb2+1;
                        BB(i,j)=1;
                    end
                end
            end
        end
        bbb1=bbb1/T;bbb2=bbb2/T;% ˮ���룬�����ҵĸ�����
        BBB = [bbb1; bbb2];
        
        %% ��������쳬���˼��޼ƻ��������ͱ���
        
        % ����ʵ�ʳ���
        CHULI_out_real =  [CHULI_out_sby, CHULI_out_ghy ,CHULI_out_gbz];
        % ���㳬������
        jx1=0; jx2=0;jx3=0;
        for j=1:3
            for i=1:T
                if  CHULI_out(i,j) > CHULi_max(i,j)
                    if j==1
                        jx1=jx1+1;
                        jx(i,j) =1;
                        CHULI_out_real (i,j) = CHULi_max(i,j);
                    elseif j==2
                        jx2=jx2+1;
                        jx(i,j) =1;
                        CHULI_out_real (i,j) = CHULi_max(i,j);
                    else
                        jx3=jx3+1;
                        jx(i,j) =1;
                        CHULI_out_real (i,j) = CHULi_max(i,j);
                    end
                end
            end
        end
        jx1=jx1/T;jx2=jx2/T;jx3=jx3/T;% ˮ���룬�����ҵĸ�����
        jxsum = [jx1; jx2; jx3];
        % �����ܺ�CHULI_out
        CHULI_out_sum1 = sum(CHULI_out);
        CHULI_out_sum = sum(CHULI_out_sum1);
        E_out_sum =  CHULI_out_sum*24/10^5;
        % �����ܺ� CHULI_out_real
        CHULI_out_real_sum1 = sum(CHULI_out_real);
        CHULI_out_real_sum = sum(CHULI_out_real_sum1);
        E_out_real_sum =  CHULI_out_real_sum*24/10^5;
        
        %% ����ʵ�ʸ���Ҫ�󣬼���ʵ�����������������ʡ�Ƿ����
        CHULI_out_net = sum(CHULI_out,2);
        CHULI_out_net_real = CHULI_out_net;
        QI_E = zeros(T,1);qi=0;
        QIAN_E = zeros(T,1);qian=0;
        for i=1:T
            if  CHULI_out_net(i,1) > CHULi_REQUIRE(i,1)
                CHULI_out_net_real(i,1) = CHULi_REQUIRE(i,1);
                QI_E(i,1) = 1;
                qi=qi+1;
            elseif CHULI_out_net(i,1) < CHULi_REQUIRE(i,1)
                CHULI_out_net_real(i,1) = CHULI_out_net(i,1);
                QIAN_E(i,1) = 1;
                qian=qian+1;
            end
        end
        qi=qi/T;qian=qian/T;
        qisum= [qi; qian];
        
        CHULI_out_net_sum = sum(CHULI_out_net);
        E_out_net_sum=  CHULI_out_net_sum*24/10^5;
        
        CHULI_out_net_real_sum = sum(CHULI_out_net_real);
        E_out_net_real_sum =  CHULI_out_net_real_sum*24/10^5;
        
        QI_QUAN = CHULI_out_net - CHULI_out_net_real;
        QI_QUAN_sum = sum(QI_QUAN);
        E_QI_QUAN = QI_QUAN_sum*24/10^5;
        
        %% ����ˮλƫ�����ǰ��ʱ��ƫ��
        Delta_Z_up = zeros(T,3);
        Delta_Q_up = zeros(T-1,3);
        for j=1:3
            for i=1:T
                Delta_Z_up(i,j) = abs(Z_up(i+1,j)- Z_up(i,j));
            end
        end
        
        for j=1:3
            for i=1:T-1
                Delta_Q_up(i,j) = abs(Q_up(i+1,j)- Q_up(i,j));
            end
        end
        Delta_Z_up_max = max(Delta_Z_up);
        Delta_Q_up_max = max(Delta_Q_up);
        
        %% ��Ž���������еĽ��
        temp1 = [temp1;round_number1 ,Q_in_round,  Q_iflow_round];
        temp2 = [temp2; round_number1 ,QQmaxmin'];
        temp3 = [temp3; round_number1 ,CHULi_max];
        temp4 = [temp4; round_number1 ,CHULi_REQUIRE];
        temp5 = [temp5;round_number2 , Z_up];
        temp6 = [temp6;round_number1 , Q_up];
        temp6_1 = [temp6_1,  Delta_Z_up_max'];
        temp7_1 = [temp7_1,  Delta_Q_up_max'];
        temp7 = [temp7;round_number1 , Delta_Z_up];
        temp8 = [temp8;round_number3 , Delta_Q_up];
        temp9 = [temp9;round_number1 ,  N_sep];
        temp10 = [temp10;round_number1 ,  CHULI_out];%���Ǹ���ˮ��վ�������
        temp11 = [temp11;round_number1 ,  CHULI_out_real]; %���Ǽ��޼ƻ��ĳ������
        temp12 = [temp12;round_number1 , CHULI_out_net];  %�����ݼ��������
        temp13 = [temp13;round_number1 , CHULI_out_net_real];
        temp14 = [temp14;round_number1 , jx];
        temp15 = [temp15;round_number1 , QI_E];
        temp16 = [temp16;round_number1 , QIAN_E];
        temp17 = [temp17;round_number1 ,  QI_QUAN];
        temp18 = [temp18; round_number1 , QW_waste];
        temp19 = [temp19; round_number1 , QW_QJ];
        
        tempR1 = [tempR1,  BBB];
        tempR2 = [tempR2,  AAA];
        tempR3 = [tempR3,  jxsum];
        tempR4 = [tempR4,  qisum];
        tempR5 = [tempR5, QW_FinSum'];
        tempR6 = [tempR6, QW_FinSum_all];
        tempR7 = [tempR7, N_FinSum];
        tempR8 = [tempR8, CHULI_out_sum1'];
        tempR9 = [tempR9, CHULI_out_sum];
        tempR10 = [tempR10, E_out_sum];
        tempR11 = [tempR11, CHULI_out_real_sum1'];
        tempR12 = [tempR12, CHULI_out_real_sum];
        tempR13 = [tempR13, E_out_real_sum];
        tempR14 = [tempR14, CHULI_out_net_sum ];
        tempR15 = [tempR15, E_out_net_sum];
        tempR16 = [tempR16, CHULI_out_net_real_sum];
        tempR17 = [tempR17, E_out_net_real_sum];
        tempR18 = [tempR18, QI_QUAN_sum];
        tempR19 = [tempR19, E_QI_QUAN];
        
        file = ([num2str(afa),'-final--0306--60days-',num2str(round),'.mat' ]);
        save (file);
        
    end
    file = ([num2str(afa),'--final--0306--60days','.mat' ]);
    save (file);
    
end
toc