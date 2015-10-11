function nl=NegLogLikelihood(fDebug, fNewCluster, Yn, a0, b0, mu0, c0)
% function to compute -logp(xn| ) for different models
%
%   Free to user under the GPL licence v3.0
%
[N,D] = size(Yn);

if(length(mu0) ~= D)
    error('prior mean should be D-dimensional');
end
        
debugThreshold = 1e-3; % under debug will trigger error message if difference greater than this for negative log analytic pdf and our calculation.

% new cluster
if(fNewCluster)
    if(length(b0) == 1)
        b0 = ones(1,D)*b0; % turn it to a row vector
    end

    % Hybrid MoG or MoS - for new cluster it's a student-t       
    nl = D/2 * log(2*pi) - D/2 * log(c0/(c0+1)) ...
        + D*gammaln(a0) - D*gammaln(a0 + 0.5) ...
        + 1/2 * sum(log(b0)) + (a0+1/2) * sum( log( 1 + (Yn - repmat(mu0,N,1)).^2 .* a0*c0./(repmat(b0,N,1).*(c0+1))./(2*a0) ) , 2);              

    if(any(isnan(nl) | ~isreal(nl) | isinf(nl)))
        error('Bad neg log likelihood %g',nl);
    end
    if(fDebug)
        L0 = a0*c0/(b0*(c0+1));  % constant for all dimensions          

        analyticPdf = 1;
        for d=1:D
            pd = makedist('tLocationScale','mu',mu0(d),'sigma',sqrt(1./L0),'nu',2*a0);
            analyticPdf = analyticPdf .* pdf(pd, Yn(:,d) );
        end
        diff = abs(-log(analyticPdf) - nl);
        if(diff > debugThreshold)
            error('check calculation - diff %.3f', diff);
        end
    end        
else
    % Student-T - passed in values are not from prior but are
    % interpreted as posterior NG parameters a_k, c_k, bk,d, m_k,d
    if(length(b0) ~= D)
        error('passing in prior as posterior NG parameter?');
    end
    nl = D/2 * log(2*pi) - D/2 * log(c0/(c0+1)) ...
        + D*gammaln(a0) - D*gammaln(a0 + 0.5) ...
        + 1/2 * sum(log(b0)) + (a0+1/2) * sum( log( 1 + (Yn - repmat(mu0,N,1)).^2 .* a0*c0./(repmat(b0,N,1).*(c0+1))./(2*a0) ) , 2);              

    if(fDebug)                          
        analyticPdf = 1;
        for d=1:D
            Ld = a0*c0/(b0(d)*(c0+1));
            pd = makedist('tLocationScale','mu',mu0(d),'sigma',sqrt(1./Ld),'nu',2*a0);
            analyticPdf = analyticPdf * pdf(pd, Yn(:,d) );
        end
        diff = abs(-log(analyticPdf) - nl);
        if(any(diff > debugThreshold))
            error('check calculation'); 
        end        
    end     
end

CheckNumber(nl);

