function [predictiveDensityResult, GeneratedSamples] =predictiveDensity(K, gamma, mu0,a0,b0,c0, YnewVector, xIndicator, Y, fGenerateSamples, NSamples)            
% Compute predictive density for each point in YnewVector and potentially
% generate samples
%
% parameters to pass in 
% N,    number of points used in inference
% K,    number of clusters
% xIndicator, indicators (size N X 1)
% gamma,    concentration parameter 
% mu0,a0,b0,c0, NG prior terms 
% Y     observations used in inference 
% YnewVector observations to predict density on
%
%   Free to user under the GPL licence v3.0
%
[N, D] = size(Y);
[n, Dn] = size(YnewVector);
if(D ~= Dn)
    error('Dimension of training and test data must match %g-%g',D,Dn);
end

% for each existing cluster precompute bk
if(length(xIndicator) ~= N)
    error('Wrong size indicators %g', length(xIndicator));
end
    
% Generate responsibilities
pgen = nan(K+1,1);
for k=1:K
        i = (xIndicator == k); 
        Nk = sum(i); % Nk does not include new point
        pgen(k) = Nk/(gamma+N);  % N without new point
end
% new cluster
pgen(K+1) = gamma/(gamma+N);

% Generating samples from mixture of Student-t's?
if(exist('fGenerateSamples','var') && fGenerateSamples)
    if(NSamples < 1)
        error('Cannot generate just %g samples.', NSamples);
    end    
    % figure out how many samples each cluster will generate
    
    [dum,compon] = histc(rand(NSamples,1), [0; cumsum(pgen(:))./sum(pgen)]); %#ok<ASGLU> % shouldn't have to sum over to ensure it normalises to 1 but just to be safe
    NSamplesCluster = nan(K+1,1);
    for k=1:(K+1)
        NSamplesCluster(k) = sum(compon == k);
    end
    if(sum(NSamplesCluster) ~= NSamples)
        error('something went wrong %g-%g', sum(NSamplesCluster), NSamples);
    end
    GeneratedSamples = nan(NSamples,D);
    
else
    fGenerateSamples = 0;    
end

if(~isempty(YnewVector))
    % Calculate the pdf
    fCalculatePdf = 1;   
    
    if(D ~= Dn)
        error('Dimension of training and test data must match %g-%g',D,Dn);
    end
else
    fCalculatePdf = 0;
end
if(fCalculatePdf == 0 && fGenerateSamples == 0)
    error('nothing to do');
end


clusterMarginal = nan(K+1,n);

% Go through all existing clusters
for k=1:K
        i = (xIndicator == k); 
        Nk = sum(i); % Nk does not include new point
        if(Nk == 0)
            continue;
        end  
        if(Nk == 1)
            sumYk = Y(i,:); % single point - no sum
        else
            sumYk = sum(Y(i,:));  % sum across columns-dimensions   
        end
        
        pi_k = pgen(k); % N without new point
        marginalK = 1;
        for d=1:D            
            priorTerm = (c0 * Nk * (sumYk(d)/Nk - mu0(d)).^2) / (2*(c0 + Nk));
            bk = (b0 + 0.5*sum( (Y(i,d)-sumYk(d)/Nk).^2 ) + priorTerm); 

            mu_n = (c0*mu0(d) + sumYk(d)) / (c0 + Nk); % careful must only include sumY - not new observation
            an = a0 + Nk/2;
            cn = c0 + Nk;
            sigmaT = sqrt(bk*(cn+1)/(an*cn));

            pd = makedist('tLocationScale','mu',mu_n,'sigma',sigmaT,'nu',2*an);
                       
            if(fGenerateSamples && NSamplesCluster(k) > 0)
                % Generate random samples from Student-t                
                if(k==1)
                    strIdx = 1;
                else
                    strIdx = 1+sum(NSamplesCluster(1:(k-1)));
                end
                if(~isnan( GeneratedSamples(strIdx,d) ))
                    error('will overwrite sample and mess up distribution of mixture');
                end
                    
                idx = strIdx:(strIdx+NSamplesCluster(k)-1);
                if(length(idx) ~= NSamplesCluster(k))
                    error('check your index');
                end
                GeneratedSamples(idx,d) = random(pd, NSamplesCluster(k), 1);
                CheckNumber(GeneratedSamples(idx,d));
            end
            
            if(fCalculatePdf)  
                % calculate pdf
                pdfValueMatlab = pdf(pd,YnewVector(:,d));                           
                marginalK = marginalK .* pdfValueMatlab;     
            end
        end
        clusterMarginal(k,:) = pi_k * marginalK;            
        CheckNumber(clusterMarginal(k,:));         
end

% new cluster
pi_kp1 = pgen(K+1);
    
marginalK = 1;
sigmaT = sqrt(b0*(c0+1)/(a0*c0));
for d=1:D   
    pd = makedist('tLocationScale','mu',mu0(d),'sigma',sigmaT,'nu',2*a0);
    pdfValueMatlab = pdf(pd,YnewVector(:,d));
    marginalK = marginalK .* pdfValueMatlab;            
    
    if(fGenerateSamples && NSamplesCluster(K+1) > 0)
        strIdx = 1+sum(NSamplesCluster(1:K));
        
        if(~isnan( GeneratedSamples(strIdx,d) ))
            error('will overwrite sample and mess up distribution of mixture');
        end

        idx = strIdx:(strIdx+NSamplesCluster(K+1)-1);
        if(length(idx) ~= NSamplesCluster(K+1))
            error('check your index');
        end        
        GeneratedSamples(idx,d) = random(pd, NSamplesCluster(K+1), 1);  
        
        CheckNumber(GeneratedSamples(idx,d));
    end                    

end
clusterMarginal(K+1,:) = pi_kp1 .* marginalK;               
CheckNumber(clusterMarginal(K+1,:));

predictiveDensityResult = nansum(clusterMarginal)'; % ignoring empty clusters, produce column vector
if(fCalculatePdf) % don't make sense to check if we are not calculating pdf 
    if(length(predictiveDensityResult) ~= n)
        error('produced incorrect length list %g', length(predictiveDensityResult));
    end
    CheckNumber(predictiveDensityResult);
end
