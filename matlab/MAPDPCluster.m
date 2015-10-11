function [Kr,xr, NLL, pLastStep, Keff, rConv, alpha0Vector] = MAPDPCluster(fDebug, R, D, alpha0, mu0,a0,b0,c0, thresholdNLLConvergence, Y, fGibbs) 
% MAP or Gibbs DP with block updates for cluster components.
% Parameters
% fDebug                Print to screen log messages?
% R                     maximum number of iterations
% D                     Dimensionality of data
% alpha0,                concentration parameter, if -1 we learn MAP value
% using posterior derived in Rasmussen (2000).
% mu0,a0,b0,c0,         NG prior parameters
% Y                     data
%
% thresholdNLLConvergence - threshold for NLL convergence check
%
%
% Returns
% Kr,                   number of clusters for each iteration
% xr,                   indicators for each iteration
% mur,taur,             mean, precisions for each cluster per iterion
% NLL,                  Negative log likelihood per iteration
% pLastStep, Keff       Nk/N and effective number of clusters at last step
% rConv                 step we converged on
%
%   Free to user under the GPL licence v3.0

if(fDebug~=0 && fDebug ~= 1)
    error('debug should be 0 or 1');
end
        
fDebugNLL = 0; % should we check NegLogLikelihood calculations? Can be quite slow for studen-t pdf


if(fGibbs == 0)
    fprintf('MAP estimation.\n');
elseif(fGibbs == 1)
    fprintf('Gibbs estimation\n');
else
    error('MAP or Gibbs?');
end

    
if(exist('thresholdNLLConvergence','var') && ~isnan(thresholdNLLConvergence) )
    CheckNumber(thresholdNLLConvergence);
    if(thresholdNLLConvergence < 0 || thresholdNLLConvergence > 100)
        error('check threshold %g', thresholdNLLConvergence);
    end
    if(length(thresholdNLLConvergence) ~= 1)
        error('check threshold dimension %g', length(thresholdNLLConvergence));
    end
else
    thresholdNLLConvergence= 1e3*eps;
end
 
[N, Dcheck] = size(Y);
if(Dcheck ~= D)
    error('Check dimension %g-%g',Dcheck,D);
end

if(length(b0) == 1)
    b0 = ones(1,D)*b0; % turn it to a row vector
end

if(alpha0 < 0)
    % we will learn MAP value for alpha0
    fLearnMAPAlpha = 1;
    % take absolute value as initial value
    alpha0 = abs(alpha0);
    if(fDebug); fprintf('MAPDPClusterMoGOrMoS:: Learning alpha0, initial value %g.\n', alpha0); end
    % error('check concentration parameter %g', alpha0);
else
    % By default we use fixed value passed in
    fLearnMAPAlpha = 0;
end

if(size(mu0,2) ~= D && size(mu0,1) ~= 1)
    error('mu0 should be row vector');
end


% initialisation 
Kr = nan(R+1,1); % number of components
Keff = nan(R+1,1); % number of components excl empty clusters
xr = nan(R+1,N); % indicators
NLL = nan(R+1,1); % Negative log likehood
emptyClusters = 0; % if 1 cluster is empty and will never be touched again
Kr(1) = 1;      % one cluster
xr(1,:) = 1;    % everybody belongs there
alpha0Vector = nan(R+1,1); % concentration parameter

alpha0Vector(1) = alpha0;

% Compute initial likelihood for one big cluster 
i = (xr(1,:) == 1); 
Nk = sum(i);

if(Nk ~= N)
    error('I assume on big cluster for initialisation');
end        
NLL(1) = sum( NegLogLikelihood(fDebugNLL, 1, Y, a0, b0, mu0, c0 ) ) - gammaln(alpha0) + gammaln(alpha0+N) - gammaln(N) - log(alpha0); % initially we have one big cluster 

if(fDebug); fprintf('Initial neg log marginal likelihood using 1 big cluster %.1f.\n', NLL(1)); end


for r = 1:R        
    x = xr(r,:);    % Current step indicators
    K = Kr(r);
    alpha0 = alpha0Vector(r);
        
    NLL(r+1) = 0;
    idxPointsPermuted = randperm(N);
    
    % As K can increase we need to reallocate this array every loop
    emptyClustersOld = emptyClusters;
    emptyClusters = zeros( max(2*N, 2*K),1);
    emptyClusters(1:length(emptyClustersOld)) = emptyClustersOld;
    
    for n=idxPointsPermuted
        % for each data point
        
        % Calculate neg log prob over component indicators/new component
        dk = nan(K+1,1);
        negLogMarginal = nan(K+1,1); 
        
        idxFullClusters = find(emptyClusters(1:K) ~= 1);
        
        for ki=1:length(idxFullClusters)
            k = idxFullClusters(ki);
            i = (x == k);
            iNotn = i;
            Nkni = sum(i);
            
            if (x(n) == k)
                Nkni = Nkni - 1; 
                iNotn(n) = 0; % turn off current point
            end                     
            if(Nkni <= 0)
                dk(k) = Inf; 
                emptyClusters(k) = 1; % one empty always empty
            else
                % mixture of student-t - need to compute NG posterior
                % without current point
                ck = c0 + Nkni;
                ak = a0 + Nkni/2;
                if(Nkni == 1)
                    % just single point
                    sumYk = Y(iNotn,:); % single point
                else 
                    sumYk = sum(Y(iNotn,:)); % sum across columns-dimensions        
                end

                mkd = (c0 * mu0 + sumYk) / ck;
                bkd = b0 + 0.5 * sum( (Y(iNotn,:) - repmat(sumYk/Nkni, Nkni, 1) ).^2 ) + c0*Nkni*(sumYk/Nkni - mu0).^2/(2*ck);
                negLogMarginal(k) = NegLogLikelihood(fDebugNLL, 0, Y(n,:), ak, bkd, mkd, ck);
                
                if(r == 1 &&  k == 1) 
                    % first cluster initialised to be full cluster - eliminate
                    % reinforcment effect by setting log(Nkn) = 0  
                    Nkni = 1; 
                end                   
                dk(k) = negLogMarginal(k) - log(Nkni);                                 
            end
        end
        % new cluster     
        negLogMarginal(K+1) =  NegLogLikelihood(fDebugNLL, 1, Y(n,:), a0, b0, mu0, c0);
        dk(K+1) = negLogMarginal(K+1) - log(alpha0);
        
        if(fGibbs)
            pxy = exp(-dk);
            pxy(emptyClusters == 1) = 0; % 0 assignment probs            
            pxy = pxy/sum(pxy);                   % Normalize
            if(any(isnan(pxy)))
                error('nan pxy');
            end
            % Sample indicator
            [~,dkWinIndex] = histc(rand,[0 cumsum(pxy,1)']);    % The 'meat' of randsample ...         
            if(dkWinIndex<1 || dkWinIndex>K+1)
                error('check %g',dkWinIndex);
            end
        else
            % MAP
            % Pick indicator with minimum value
            [dkMinValue, dkWinIndex] = min(dk);

            if(length(dkWinIndex) ~= 1)
                error('%g winners! should be 1. Minimum dK value if %g', dkWinIndex, dkMinValue);
            end      
        end
        x(n) = dkWinIndex; % we update local state
                
        % Update component count 
        if (x(n) > K)            
            K = K + 1;
            if(fDebug); fprintf('%g: Increment number of clusters to %g.\n',r,K); end
            i = (x == x(n)); 
            Nk = sum(i);        % Need to update the cluster count even for MoS so NLL calculation below is correct
            if(Nk ~= 1)
                error('New cluster should have exactly one observation assigned to it! It has %g.',Nk);
            end
            
        end
        
        % Negative complete log likelihood (NLL) for MoS      
        NLL(r+1) = NLL(r+1) + negLogMarginal(dkWinIndex); 
        CheckNumber(NLL(r+1));                
    end          
    
    % Calculate p (mixture components)
    p = histc(x,1:K)/N;
    
    pLastStep = p;
    
    Keff(r+1) = K - sum(p==0);    
    
    % Update concentratin parameter if asked, using Newton's method
    if(fLearnMAPAlpha)
        if(fGibbs)
            error('cannot learn alpha in gibbs - not implemented in code. Only MAP is.');
        end
        options = optimoptions(@fminunc,'GradObj','on','Hessian','on','Display','off');
        a_conc = 5; b_conc = .1;
        alpha0 = exp(fminunc(@CalcAlpha0Posterior,log(alpha0), options ,N, Keff(r+1), a_conc, b_conc)); % we use Keff - not K!

        if(fDebug); fprintf('alpha0 -> %g\n', alpha0); end
    end
    
    % update state variables for next step
    xr(r+1,:) = x;
    Kr(r+1) = K;
    alpha0Vector(r+1) = alpha0;
        
    % Update NLL with concentration parameter effect and reinforcment
    % effect
    idxFullClusters = find(emptyClusters(1:K) ~= 1);
    logGammaNk = 0;
    for ki=1:length(idxFullClusters)
        k = idxFullClusters(ki);

        i = (x == k); 
        Nk = sum(i);    
        if(Nk > 0)
            logGammaNk = logGammaNk + gammaln(Nk);
        end
    end
    NLL(r+1) = NLL(r+1) - gammaln(alpha0) + gammaln(alpha0+N) - logGammaNk - Keff(r+1)*log(alpha0); 
    
    fprintf('%g: MAP Gibbs NLL %.2f: K=%d. Effective K %g. \n',r, NLL(r+1),K, Keff(r+1)); 
    if(fDebug)        
        
        for ip=1:length(p)
            if(p(ip)~=0)
                fprintf('Probs: %0.2f\n',p(ip));
            end
        end
        if(sum(p==0)>0)
            fprintf('Empty clusters: %g.\n', sum(p==0));
        end    
    end
    
    if(fGibbs)
        % Gibbs convergence
        if(r > R) 
            % Could use raftery diagnostic, available in the econometrics
            % toolbox - see http://www.spatial-econometrics.com/
            % res = raftery(NLL(2:r+1),0.025,0.01,0.95);  %          
             rConv = r+1;
            break;
        end                 
    elseif(r>1 && abs(NLL(r+1) - NLL(r)) < thresholdNLLConvergence) 
        if(fDebug); fprintf('MAPDP Algorithm converged after %g steps.\n',r); end
        rConv = r+1;        
        break;
    end
    if(fDebug); disp('---------------------------'); end
end
    
if(r == R)
    rConv = R + 1;
end
