%%inverse material design based on BP ANN
%% clean calculation space
clc
clear
close all

%% reading, division and normalization of training data and testing data
%%reading input data and corresponding output data
[data]=xlsread('Mplist.xlsx',1);

Out=linspace(0,data(1,5),100);
p=polyfit(data(1,:),data(2,:),3);
In(1,:)=polyval(p,Out);

p=polyfit(data(1,:),data(3,:),4);
In(2,:)=polyval(p,Out);

p=polyfit(data(1,:),data(4,:),4);
In(3,:)=polyval(p,Out);

p=polyfit(data(1,:),data(5,:),4);
In(4,:)=polyval(p,Out);

p=polyfit(data(1,:),data(6,:),3);
In(5,:)=polyval(p,Out);

figure(1)
subplot(2,3,1)
plot(Out,In(1,:),'-b',data(1,:),data(2,:),'or','Markersize',7,'Markerface','white','linewidth',2.0);
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('Mode')
xlabel('?_1')
subplot(2,3,2)
plot(Out,In(2,:),'-b',data(1,:),data(3,:),'+r','Markersize',7,'Markerface','white','linewidth',2.0);
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('Stiffness(MPa)')
xlabel('?_1')
subplot(2,3,3)
plot(Out,In(3,:),'-b',data(1,:),data(4,:),'sr','Markersize',7,'Markerface','white','linewidth',2.0);
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('?c(kN/m)')
xlabel('?_1')
subplot(2,3,4)
plot(Out,In(4,:),'-b',data(1,:),data(5,:),'*r','Markersize',7,'Markerface','white','linewidth',2.0);
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('?p(kN/m)')
xlabel('?_1')
subplot(2,3,5)
plot(Out,In(5,:),'-b',data(1,:),data(6,:),'^r','Markersize',7,'Markerface','white','linewidth',2.0);
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('Resilience(J.m?2)')
xlabel('?_1')

%divide data into training cases and testing data
%randomly divided data sets
[m,n]=size(Out);
k=rand(1,n);
[M,N]=sort(k);
testdIn=In(:,N(1:20));%%pick out input and output for testing
testdOut=Out(N(1:20));
traindIn=In(:,N(21:n));%%pick out input and output for training
traindOut=Out(N(21:n));

%training data normalization
[train_inN,tranin_inS]=mapminmax(traindIn);
[train_outN,train_outS]=mapminmax(traindOut);

%% BP network training
%Initialization
net=newff(train_inN,train_outN,[30,20]);

net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00005;

%Training
net=train(net,train_inN,train_outN);

%% BP network inverse design
%testing data normalization
test_inN=mapminmax('apply',testdIn,tranin_inS);

%classification
Ivd=sim(net,test_inN);

%%anti-classification
FinIVD=mapminmax('reverse',Ivd,train_outS);

%% Result analysis
%classified a_e and exact
[m,n]=size(FinIVD);
figure(2)
plot(1:1:n,FinIVD(1,:),':ob','Markersize',7,'Markerface','white','linewidth',3.0)
hold on
plot(1:1:n,testdOut(1,:),'-k','Markersize',7,'Markerface','white','linewidth',1.0);
hold on
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
legend('Inver-design','Expe-design')
ylabel('')
xlabel('Testing NO.')

%prediction error
error=FinIVD-testdOut;

figure(3)
plot(1:n,error,'-o','Markersize',7,'Markerface','white','linewidth',2.0)
hold on
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('error')
xlabel('Testing NO.')

figure(4)
plot((testdOut(1,:)-FinIVD(1,:))./testdOut(1,:),'-o','Markersize',7,'Markerface','white','linewidth',2.0);
hold on
set(gca,'FontName','Times New Roman', 'FontSize', 24)
set(get(gca,'XLabel'),'FontName','Times New Roman','Fontsize',24)
set(get(gca,'YLabel'),'FontName','Times New Roman','Fontsize',24)
ylabel('error percentage')
xlabel('Testing NO.')

Aveerror=mean(error);
Sumerror=sum(error);