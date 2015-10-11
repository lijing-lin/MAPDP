%
% 1-D simple example to test MAP DP clustering algorithms
%
%   Free to user under the GPL licence v3.0
%
% Tested on MATLAB version 8.2.0.701 (R2013b)
%
clear all; close all;
%rng(5489,'twister'); For newer versions of matlab
rand('seed',352)
randn('seed',532)

% Model parameters
% Prior Dirichlet concentration parameter
% set negative value to learn it by numerically minimising conditional
% posterior (see supplementary material in paper)
alpha0 = 9;
fDebug = 1; % will print messages during optimisation

N=1000; % data set size

D = 1;  % dimensionality of data

Kgen = 3; % how many clusters in generative GMM model

% Maximum number of iterations
R = 100;                % Number of iterations

%% Generate data

if(Kgen == 2)
    sigmagen(1) = 1.6; % variances
    sigmagen(2) = 0.9;
    mugen = [2 -20];
    pigen = 0.8;
    pgen = [1-pigen pigen];
elseif(Kgen == 3)
    sigmagen(1) = 3; % variances
    sigmagen(2) = 0.9;
    sigmagen(3) = 5;
    mugen = [14 -20 20];
    pgen = [0.4 0.3 0.3];
else
    error('Check');
end

% Generate categorical indicator (x) data
trueLabels = randsample(Kgen,N,true,pgen);

% Generate Gaussian observation (y) data
Y = zeros(N,D);
YTest = zeros(N,D);

for k = 1:Kgen
    i = find(trueLabels == k);
    M = length(i);
    Y(i,:) = sqrt(sigmagen(k))*randn(1,M)+repmat(mugen(k),1,M);
    YTest(i,:) = sqrt(sigmagen(k))*randn(1,M)+repmat(mugen(k),1,M); % can use same trueLables    
end


mu0 = mean(Y);
c0 = 10/length(Y); 
a0 = 1; 
b0 = var(Y)/30;


%% run code    
fGibbs = 0; % 1==Gibbs MCMC, 0==MAP
[Kr,xr,NLL, pLastStep, Keff, rConv, alpha0Vector] = MAPDPCluster(fDebug, R, D, alpha0, mu0,a0,b0,c0, nan, Y, fGibbs);

alpha0 = alpha0Vector(rConv);

% Compare results
fprintf('Keff = %g \n', Keff(rConv));

disp('indicator probabilities true');
pgen
disp('indicate probabilities estimated');
idxNonEmpty = (pLastStep~=0);
pLastStep(idxNonEmpty)


%% Plot results
close all;
figure;
subplot(3,1,1);
hold on;
n = hist(xr(1:rConv,:)',1:Kr(rConv))/N;
plot(n');
plot([1 rConv],[pgen' pgen']',':');
ylim([-0.1 1.1]);
xlabel('Iteration r');
ylabel('Indicator prob');

subplot(3,1,2);
plot(Kr); hold on;
ylim([-0.1 max(Kr)+0.1]);
plot([1 rConv], [Kgen Kgen], ':');
xlabel('iteration r');
ylabel('Components K');

subplot(3,1,3);
plot(NLL(1:rConv));
xlabel('iteration');
ylabel('NLL');



%% Calculate predictive density
YnewVector = linspace(min(Y), max(Y), 1000)';
 
[predictiveDensityResult, samples]=predictiveDensity(Kr(rConv), alpha0, mu0,a0,b0,c0, YnewVector, xr(rConv,:), Y,1,100);
    

figure; hold on;
sigmagen_1(1,1,1:Kgen) = sigmagen; % variances
truegmm = gmdistribution(mugen',sigmagen_1,pgen); % pass in variances, i.e. covariance matrix diagonal
pdfGenerativeGMM = pdf(truegmm, YnewVector);
plot(YnewVector, pdfGenerativeGMM, '--b');
plot(YnewVector, predictiveDensityResult, '-.m');
v=axis;
plot(samples,0.01, '+');
axis(v);
legend('true distribution','prediction','samples');


