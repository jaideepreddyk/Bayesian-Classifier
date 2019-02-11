M1=[];
S1=[];
M2=[];
S2=[];
%%%%%%%%%%%%%%%%%  Step 1. Training  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getting the mean and standard deviation using top 100 subjects
for i=1:5
    M1(i)=mean(F1(1:100,i));
    M2(i)=mean(F2(1:100,i));
    S1(i)=std(F1(1:100,i));
    S2(i)=std(F2(1:100,i));
end
%%%%%%%%%%%%%%%%%%%%%%%% Step 2.1 Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%

P=[];
A2=zeros(900,5);

for j=101:1000
    for i=1:5
        x=F1(j,i)
      for k=1:5
        P(k)=normpdf(x,M1(k),S1(k));
      end
    [val,index]=max(P);
    A2(j-100,i)=index;
    end
end
    
%%%%%%%%%%%%%%%%%%%%%%%% Step 2.2 Accuracy %%%%%%%%%%%%%%%%%%%%%%%%%%%
 
b=0;
e=0;
for i=1:5
    for j=1:900
        if A2(j,i)==i
            b=b+1;
        else 
            e=e+1;
        end
    end
end

acc=b/4500
err=e/4500

%%%%%%%%%%%%%%%%%%%%%%% Step 3 Standard Normal %%%%%%%%%%%%%%%%%%%
% computing Z1
Z1=zeros(1000,5)
row_m=mean(F1,2);
row_std=std(F1,0,2);
for i=1:1000
    for j=1:5
        Z1(i,j)=(F1(i,j)-row_m(i))/row_std(i);
    end
end

% Plotting F1 vs F2
plot(F1(1:1000,1),F2(1:1000,1),'o','color','red')
hold on;
plot(F1(1:1000,2),F2(1:1000,2),'o','color','black')
hold on;
plot(F1(1:1000,3),F2(1:1000,3),'o','color','green')
hold on;
plot(F1(1:1000,4),F2(1:1000,4),'o','color','blue')
hold on;
plot(F1(1:1000,5),F2(1:1000,5),'o','color','magenta')
hold on;
title("Distribution of data using F1 and F2")

figure()
% Plotting Z1 vs F2
plot(Z1(1:1000,1),F2(1:1000,1),'o','color','red')
hold on;
plot(Z1(1:1000,2),F2(1:1000,2),'o','color','black')
hold on;
plot(Z1(1:1000,3),F2(1:1000,3),'o','color','green')
hold on;
plot(Z1(1:1000,4),F2(1:1000,4),'o','color','blue')
hold on;
plot(Z1(1:1000,5),F2(1:1000,5),'o','color','magenta')
hold on;
title("Distribution of data using Z1 and F2")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Step 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% CASE 2 X=Z1 %%%%%%%%%%%

% Training
for i=1:5
    MZ1(i)=mean(Z1(1:100,i));
    SZ1(i)=std(Z1(1:100,i));
end
   
 %Testing

P=[];
A2=zeros(900,5);

for j=101:1000
    for i=1:5
        x=Z1(j,i)
      for k=1:5
        P(k)=normpdf(x,MZ1(k),SZ1(k));
      end
    [val,index]=max(P);
    A2(j-100,i)=index;
    end
end

%Accuracy
b=0;
e=0;
for i=1:5
    for j=1:900
        if A2(j,i)==i
            b=b+1;
        else 
            e=e+1;
        end
    end
end

accz1=b/4500
errz1=e/4500

%%%%%%%% Case 3 X=F2 %%%%%%%%%%

%Testing
P=[];
A2=zeros(900,5);

for j=101:1000
    for i=1:5
        x=F2(j,i)
      for k=1:5
        P(k)=normpdf(x,M2(k),S2(k));
      end
    [val,index]=max(P);
    A2(j-100,i)=index;
    end
end

%Accuracy

b=0;
e=0;
for i=1:5
    for j=1:900
        if A2(j,i)==i
            b=b+1;
        else 
            e=e+1;
        end
    end
end

acc2=b/4500
err2=e/4500

%%%%%% Case 4 Multivariate Normal Distribution %%%%%%%%%
MV_mean(:,1)=MZ1;
MV_mean(:,2)=M2;
MV_cov=zeros(2,2,5);
for i=1:5
    MV_cov(:,:,i)=cov(Z1(1:100,i),F2(1:100,i));
end

% Testing
P=[];
A_MV=zeros(900,5);

for j=101:1000
    for i=1:5
        x=Z1(j,i)
        y=F2(j,i)
      for k=1:5
        P(k)=mvnpdf([x y],MV_mean(k,:),MV_cov(:,:,k));
      end
    [val,index]=max(P);
    A_MV(j-100,i)=index;
    end
end

%Accuracy
b=0;
e=0;
for i=1:5
    for j=1:900
        if A_MV(j,i)==i
            b=b+1;
        else 
            e=e+1;
        end
    end
end

% Multivariate Classification Accuracy and Error rate
mv_acc=b/4500
mv_err=e/4500

acc,accz1,acc2,mv_acc

fprintf("Classification rate of F1 : %d \n",acc)
fprintf("Classification rate of Z1 : %d \n",accz1)
fprintf("Classification rate of F2 : %d \n",acc2)
fprintf("Classification rate of Multivariate normal Z1 and F2 : %d \n",mv_acc)




