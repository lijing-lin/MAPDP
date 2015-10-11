function [lnfa,dlnfa,d2lnfa] = CalcAlpha0Posterior(y,N,K,a_g,b_g)
% Function to calculate alpha0 conditional posterior
% a_g,b_g are the Gamma prior parameters for the concentration
% parameter
%
%   Free to user under the GPL licence v3.0
%
alpha = exp(y); % concentration parameter

lnfa  = gammaln(alpha) - gammaln(alpha + N) + (K+a_g-1).*log(alpha) - b_g .* alpha;

dlnfa = alpha.*psi(alpha) - alpha.*psi(alpha+N) + (K+a_g-1) - b_g .* alpha; % derivative w/respect to log(a)

d2lnfa = alpha .* psi(alpha) + alpha.^2 .* psi(1,alpha) - alpha .* psi(alpha+N) - alpha.^2 .* psi(1,alpha+N) - b_g .* alpha;

% minimize so flip it
lnfa = -lnfa;
dlnfa = -dlnfa;
d2lnfa = -d2lnfa;
    