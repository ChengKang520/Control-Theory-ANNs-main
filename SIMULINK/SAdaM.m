%函数的总入口，收到信号后，首先进入这个函数.这个函数包含一个switch语句，根据情况进入不同的子函数
function [sys,x0,str,ts] = SAdaM(t,x,u,flag)
switch flag, 
  case 0,
    [sys,x0,str,ts]=mdlInitializeSizes; 
  case 1,
    sys=mdlDerivatives(t,x,u); 
  case 2,
    sys=mdlUpdate(t,x,u); 
  case 3,
    sys=mdlOutputs(t,x,u); 
  case 4,
    sys=mdlGetTimeOfNextVarHit(t,x,u);  
  case 9,
    sys=mdlTerminate(t,x,u);
 
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
 
end
 
%S-function进行基本的设置，相当于构造函数，定义S函数的基本特性，包括采样时间、连续或者离散状态的初始条件和Sizes数组
function [sys,x0,str,ts]=mdlInitializeSizes
sizes = simsizes;              %调用构造函数，生成一个默认类
sizes.NumContStates  = 0;      %设置系统连续状态的数量,如果是2个输出，则设为2
sizes.NumDiscStates  = 5;      %设置系统离散状态的数量
sizes.NumOutputs     = 1;      %设置系统输出的数量，如果是2个输出，则设为2
sizes.NumInputs      = 2;      %设置系统输入的数量
sizes.DirFeedthrough = 1;      %设置系统直接通过量的数量，一般为1
sizes.NumSampleTimes = 1;      % At least one sample time is needed
                               % 采样时间个数，1表示只有一个采样周期.
                               % 猜测为如果为n，则下一时刻的状态需要知道前n个状态的系统状态
sys = simsizes(sizes);         %将sizes结构体中的信息传递给sys
x0  = [0;0;0;1;1];                   % 系统初始状态
% x0  = [0;0;0];                   % 系统初始状态
str = [];                      % 保留变量，保持为空
ts  = [-1 0];                   % 采样时间
 
%该函数仅在连续系统中被调用，计算连续状态变量的微分方程，求所给表达式的等号左边状态变量的积分值的过程
function sys=mdlDerivatives(t,x,u)   %Time-varying model
sys = [];
 
%sys=mdlUpdate(t,x,u); 该函数仅在离散系统中被调用，用于产生控制系统的下一个状态；更新离散状态、采样时间和主时间步的要求
function sys=mdlUpdate(t,x,u)   %更新离散状态
T=0.02;

sys=[
    u(1); %P部分-本次偏差
    x(2)+0.5*u(1)*T;%AdaM 分子的部分，偏差的累加的平方
    x(3)+0.999*u(1)*u(1)*T*T;
    x(4)+0.5*T;
    x(5)+0.999*T];  %AdaM 分母的部分，偏差的累加的平方];    

% sys=[
%     u(1); %P部分-本次偏差
%     x(2)+0.9*u(1)*T;%I部分，偏差的累加  
%     (u(1)-u(2))/T];%D部分，偏差的变化
 

%产生（传递）系统输出
function sys=mdlOutputs(t,x,u)
kp=1;
ki=5;
kd=10;

% sys=kp*x(1);  % SGD optimizer
% sys=kp*x(1)+ki*x(2);  % SGDM optimizer
% sys=kp*x(1)+ki*x(2)+kd*x(3);  %  PID optimizer

%%%%%%%%%%% AdaM optimizer 
AdaM_I=(1 / x(4)) / (sqrt(x(3) / x(5))+0.00000001)  % AdaM optimizer best
AdaM_P=(sqrt(x(5))+0.00000001)/x(4)
sys=AdaM_P*x(1) + AdaM_I*x(2);
 


% sys=mdlGetTimeOfNextVarHit(t,x,u) 获得下一次系统执行（next hit）的时间，该时间为绝对时间 此函数仅在采样时间数组中指定变量离散时间采样时间[-2 0]时会被调用。
function sys=mdlGetTimeOfNextVarHit(t,x,u) 
% sampleTime = 1;   %设置下一个采样时间在1s之后
% sys = t + sampleTime;
 sys = [];
 
% sys=mdlTerminate(t,x,u) 相当于构析函数，结束该仿真模块时被调用
function sys=mdlTerminate(t,x,u) 
sys = [];