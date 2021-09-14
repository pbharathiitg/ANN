clear all;
clc;
K = readmatrix('data_r.xlsx');
disp('no of input neurons according to data set is 8');
l = input('Enter the no of input neurons: '); %no of input neurons
m = 20; % no of hidden neurons
disp('no of output neurons according to data set is 2');
n = input('Enter the no of output neurons: ');  % no of output neurons
disp('Total number of available patterns in data set is 768');
disp('please choose the training and testing data accordingly');
trp = input('Enter the no of training patterns: ');  % no of training pattern
tep = input('Enter the no of testing patterns: ');   %no of testing patterns
P = trp; % no of training pattern
M = K([1:P],:);
N = K([(trp+1):(trp+tep)],:);
si_tr = size(M);
MAX = max(M,[],1); 
MIN = min(M,[],1);
for i = 1:si_tr(2)
    M(:,i) = 0.1 +(0.8*((M(:,i)-MIN(i))/(MAX(i)-MIN(i))));
end

I = [ones(1,P);[M(:,[1:l])]'];
TO = [M(:,[l+1:si_tr(2)])]';
size(I);
size(TO);
rng(1);
s = rng;
V = [randn(1,m);randn(l,m)];
W = [randn(1,n);randn(m,n)];
size(V);
size(W);
MSE = 1 ; %MEAN SQUARE INITIAL VALUE assume
des_err = 0.0011;
lr = 0.8; % learning rate
mt = 0.5;  % momentum
count =0;
delW_mom = 0;
delV_mom = 0;
A = W;
B = V;
disp('Running the ANN..,,');
disp('it may take a minimum of 30 mins');
while MSE>des_err
    
    for p = 1:P
        for j = 1:m
            a =0;
            for i = 1:l+1
                a = a+(I(i,p)*V(i,j));
                IH(j,p) =a;
                OH(j+1,p) = 1/(1 +exp(-a));
            end        
        end       
    end
%     IH
     for i=1:P
         OH(1,i) = I(1,i);%change required to include bias in matrix
     end                        
     OH;

    for p = 1:P
        for k = 1:n
            a =0;
            for j = 1:(m+1)
                a = a +(OH(j,p)*W(j,k));
                IO(k,p) =a;
                OO(k+1,p) = (exp(a) - exp(-a))/(exp(a) +exp(-a));
            end        
        end       
    end

     %IO
     %OO = OO>0;
     for i=1:P
          OO(1,i) = 0;%change required to include bias in matrix
     end                        
      OO;

    error = 0;
    for p = 1:P
        for k = 1:n
%             TO(k,p)
%             OO(k,p)
            er = (TO(k,p)-OO(k+1,p));
            error = error + (0.5*(er)*(er));
        end
    end
    MSE = error/P
    mse(count+1) = MSE;
    ITE_NUM(count+1) = count;


    for i=1:m+1
        for j = 1:n
            a = 0;
            for p = 1:P
    
                a = a + ((TO(j,p)-OO(j+1,p))*(1-(OO(j+1,p)*OO(j+1,p)))*OH(i,p));
                a;
            end
            
            delW(i,j) = (lr*a)/P;
        end
    end
    delW = delW + (mt*delW_mom);

    for i = 1:l+1
        for j = 1:m
            b = 0;
            for p = 1:P
                c = 0;
                for k = 1:n
                    c = c + (-(TO(k,p)-OO(k+1,p))*(1-(OO(k+1,p)*OO(k+1,p)))*W(j+1,k)*OH(j+1,p)*(1-OH(j+1,p))*I(i,p)); 
                end
                 b = b + c;  
                 delV(i,j) = ((-lr)*b)/(n*P); 
            end
        end        
    end
    delV = delV + (mt*delV_mom);
    W = W + delW ;
    V = V + delV  ;
    W;
    V;
    delW_mom= delW;
    delV_mom= delV;
    count = count +1;
    count;
end
W;
V;
final_delW = W - A;
final_delV = V - B;
for i = 1:n
 OO(i+1,:) = (((OO(i+1,:)-0.1)*(MAX(i+l)-MIN(i+l)))/0.8)+ MIN(i+l);
end



LTP = tep; %no of testing pattern
MAXtt = max(N,[],1); 
MINtt = min(N,[],1);
for i = 1:l
    N(:,i) = 0.1 +(0.8*((N(:,i)-MINtt(i))/(MAXtt(i)-MINtt(i))));
end
TI = [ones(1,LTP);([N(:,[1:l])]')];

    
    for p = 1:LTP
        for j = 1:m % no of hidden neurons
            a =0;
            for i = 1:l+1 % no of input neurons
                a = a+(TI(i,p)*V(i,j));
                IHTE(j,p) =a;
                OHTE(j+1,p) = 1/(1 +exp(-a));
            end  
            
        end       
    end
%     IH
     for i=1:LTP
         OHTE(1,i) = I(1,i);%change required to include bias in matrix
     end                        
     OHTE;

    for p = 1:LTP
        for k = 1:n  % no of output neurons
            a =0;
            for j = 1:m+1
                a = a +(OHTE(j,p)*W(j,k));
                IOTE(k,p) =a;
                OOTE(k+1,p) = (exp(a) - exp(-a))/(exp(a) +exp(-a));
            end   
            
        end       
    end

     %IO
     for i=1:LTP
          OOTE(1,i) = i;%change required to include bias in matrix
     end                        
      OOTE;

for i = 1:n
 OOTE(i+1,:) = (((OOTE(i+1,:)-0.1)*(MAXtt(i+l)-MINtt(i+l)))/0.8)+ MINtt(i+l);
end
disp('the predicted outputs for the testing patterns using ANN: ')
disp('   Y1\tY2');
output = OOTE';
output(:,[2,3])

figure(1)
plot(ITE_NUM,mse,'Linewidth',1.5);
title('MSE vs NUMBER OF ITERATIONS');
xlabel('Number of Iterations');
ylabel('Mean Square Error(MSE)');

fileID = fopen('Mean square error with Number of Iterations.txt','w');
LLL = [ITE_NUM;mse];
fprintf(fileID,'%6s %12s\n','ITERATION NO','MSE');
fprintf(fileID,'%d \t\t %.5f\n',LLL);
fclose(fileID);


out_comp=[output(:,1) N(:,l+1) output(:,2) N(:,l+2) output(:,3)];
fileID = fopen('Output of the network.txt','w');
fprintf(fileID,'%18s\t%12s\t%12s\t%12s\t%12s\n','testing pattern no','True OUTPUT 1(Y1)','Predicted OUTPUT 1(Y1)','True OUTPUT 2(Y2)','Predicted OUTPUT 2(Y2)');
fprintf(fileID,'%d\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\n',out_comp');
fclose(fileID);


for i = 1:n
    error_Y(:,i) = (((output(:,i+1)-N(:,l+i))./N(:,l+i))*100);
end

avg_error_Y = mean(error_Y);

fileID = fopen('Error in prediction.txt','w');
fprintf(fileID,'%12s\t%12s\n','Error % (Y1)','Error % (Y2)');
fprintf(fileID,'%8.4f \t %8.4f\n',error_Y');
fprintf(fileID,'\n\n%12s\t%12s\n','Aversge Error % (Y1)','Average Error % (Y2)');
fprintf(fileID,'%8.4f \t\t %8.4f\n',avg_error_Y');
fclose(fileID);

