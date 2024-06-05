
function y = Calculate_dis(Z_up,Q_in,Q_iflow,jisuanbuchang,aZV)
Q_iflow = [Q_iflow 0];
[m,n] = size(Z_up); %m为水位个数=调度期天数+1
V = zeros(m,n);
Q_dis = zeros(1,n);

for i = 1:n
ZV = aZV{i};    
    for j = 1:m
        V(j,i) = chazhi1(ZV(:,1),ZV(:,2),Z_up(j,i));
    end
    Q_dis(1,i) = (V(1,i) - V(2,i))*10^8/(jisuanbuchang*3600) + Q_in;
    Q_in = Q_dis(1,i) + Q_iflow(i);
end  
y = Q_dis;




