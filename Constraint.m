
%% ------------------- Satisfy the constraints -------------------

function y = Constraint(Q_dis,Z1,Z2,reservoir,aZD,aQmin,aZQ,ahmaxmin,aHK,jisuanbuchang,aOH,ah,aN,aNmin, QQmaxmin1,afa)

if reservoir=='shuibuya'
    iii=1;lengthofH=3;
elseif reservoir=='geheyan'
    iii=2;lengthofH=8;
else
    iii=3; lengthofH=3;
end


ZD = aZD{iii};
Qmax = chazhi1(ZD(:,1),ZD(:,2),(Z1+Z2)/2);
Qmin = aQmin{iii};
if Q_dis <= Qmax && Q_dis >= Qmin
    ZQ = aZQ{iii};
    Z_down = chazhi1(ZQ(:,2),ZQ(:,1),Q_dis);
    
    %dZ = Z1 - Z2;   %添加相邻时段水位波动差约束
    %if abs(dZ) <= 1
    
    OH=aOH(iii);
    h = (Z1 + Z2)/2 - Z_down - OH;
    hmaxmin = ahmaxmin{iii};
    if h <= hmaxmin(1) && h >= hmaxmin(2)
        HK = aHK{iii};
        K = chazhi1(HK(:,1),HK(:,2),h);
        y = K*h*Q_dis*jisuanbuchang/10^8;
        Nmin=aNmin(iii);
        Nmax = find_data2(h,ah{iii},aN{iii},lengthofH);
        if y > Nmax*1000*jisuanbuchang/10^8
            y = Nmax*1000*jisuanbuchang/10^8;
        elseif y < Nmin*1000*jisuanbuchang/10^8
            y = -0.4*10^6;
        end
    else
        y = -0.3*10^6;
    end
    
    %else
    %   y = -3*10^30;
    %end
elseif Q_dis > Qmax
    y =  -3*10^8;
elseif Q_dis > 0 && Q_dis < Qmin
    y = -3*10^7;
else
    y = -3*10^8;
end


% 设置覆盖率度区间的惩罚

% 第一组系数是 -0.95-0.05,1-20,22-26,31
% 21天，系数是0.01-0.01
% 29天，30天，系数是0.04-0.04
% % 28天，系数是0.09-0.05
if iii == 1
    if Q_dis > QQmaxmin1(1)
        y=y-afa*(Q_dis-QQmaxmin1(1))^0.5;
    end
    if Q_dis < QQmaxmin1(2)
        y=y-afa*(-Q_dis+QQmaxmin1(2))^0.5;
    end
end

if iii == 2
    if  Q_dis > QQmaxmin1(3)
        y=y-afa*(Q_dis-QQmaxmin1(3))^0.5;
    end
    if Q_dis < QQmaxmin1(4)
        y=y-afa*(-Q_dis+QQmaxmin1(4))^0.5;
    end
end

